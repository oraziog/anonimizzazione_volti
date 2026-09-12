//! Camera identity from image EXIF metadata (spec §2 camera mapping).
//!
//! With `CAMERA_ID_SOURCE=exif` the ingest path prefers the **body serial
//! number** written by the camera into each frame over the archive layout
//! (`CAM_001/foto.jpg`): serials are unique per device and survive file
//! renames, merges and SD-card dumps, so the FSM learns the real camera
//! instead of a naming convention. Any parse failure, missing tag or unusable
//! value yields `None` and the caller keeps the filename identity — a missing
//! EXIF field must never reject an otherwise valid frame (GDPR: every image
//! is still anonymized, just mapped to the filename camera).

use std::io::Cursor;

/// Extracts a camera id from the EXIF metadata of an image, or `None` when
/// no usable serial number is present (caller falls back to the filename).
///
/// The standard Exif `BodySerialNumber` (0xA431, ASCII) is the only portable
/// serial tag; vendor MakerNote serials (Nikon/Canon/Apple…) are undocumented
/// binary formats that differ per model, so they are deliberately not
/// attempted.
pub fn exif_camera_id(bytes: &[u8]) -> Option<String> {
    let reader = exif::Reader::new()
        .read_from_container(&mut Cursor::new(bytes))
        .ok()?;
    if let Some(field) = reader.get_field(exif::Tag::BodySerialNumber, exif::In::PRIMARY) {
        if let Some(id) = field_to_camera_id(field) {
            return Some(id);
        }
    }
    // Tolerant fallback: a few writers place tag 0xA431 outside the Exif
    // SubIFD (e.g. directly in IFD0); scan every parsed field by tag number.
    let by_scan = reader
        .fields()
        .find(|f| f.tag.number() == 0xA431)
        .and_then(field_to_camera_id);
    by_scan
}

/// Raw bytes of one tag value, for the types cameras actually use for
/// serials (ASCII is the standard; some bodies write Undefined/Byte).
fn field_bytes(field: &exif::Field) -> Option<Vec<u8>> {
    match &field.value {
        // ASCII stores one string per count element (cameras write a single
        // one for a serial number).
        exif::Value::Ascii(v) => Some(
            v.first()?
                .iter()
                .copied()
                .filter(|b| *b != 0)
                .collect(),
        ),
        exif::Value::Undefined(v, _) | exif::Value::Byte(v) => {
            Some(v.iter().copied().filter(|b| *b != 0).collect())
        }
        _ => None,
    }
}

fn field_to_camera_id(field: &exif::Field) -> Option<String> {
    let raw = field_bytes(field)?;
    sanitize_serial(&raw)
}

/// Sanitizes one raw serial into the camera-id grammar
/// (`[A-Za-z0-9_-]{2,64}`, must start alphanumeric — the same rules
/// `zip_worker::valid_camera_id` enforces for DB keys and crop paths).
/// Unsafe characters become `_`, then leading/trailing placeholders are
/// trimmed; a value that cannot satisfy the grammar is rejected (`None`).
fn sanitize_serial(raw: &[u8]) -> Option<String> {
    let s = String::from_utf8_lossy(raw);
    let mut id = String::new();
    for c in s.trim().chars() {
        if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
            id.push(c);
        } else {
            id.push('_');
        }
    }
    let id = id.trim_matches('_');
    if id.len() < 2 || !id.starts_with(|c: char| c.is_ascii_alphanumeric()) {
        return None;
    }
    Some(id.chars().take(64).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizes_and_caps_serials() {
        assert_eq!(sanitize_serial(b"1182050402").as_deref(), Some("1182050402"));
        assert_eq!(sanitize_serial(b" AB-99_2 ").as_deref(), Some("AB-99_2"));
        // Spaces and punctuation become placeholders, then are trimmed.
        assert_eq!(sanitize_serial(b"EOS 5D #42!").as_deref(), Some("EOS_5D__42"));
        // Grammar violations (short / non-alphanumeric start) are rejected.
        assert_eq!(sanitize_serial(b"\x01\x02").as_deref(), None);
        assert_eq!(sanitize_serial(b"").as_deref(), None);
        assert_eq!(sanitize_serial(b"_-_-").as_deref(), None);
        // Values longer than the 64-char cap are truncated, not rejected.
        let long = "a".repeat(80);
        assert_eq!(sanitize_serial(long.as_bytes()).map(|s| s.len()), Some(64));
    }

    #[test]
    fn jpeg_without_exif_yields_none() {
        use image::ImageEncoder;
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([0, 0, 0]));
        let mut buf = Cursor::new(Vec::new());
        image::codecs::jpeg::JpegEncoder::new(&mut buf)
            .write_image(img.as_raw(), 4, 4, image::ColorType::Rgb8)
            .unwrap();
        assert!(exif_camera_id(buf.get_ref()).is_none());
    }

    #[test]
    fn non_image_bytes_yield_none() {
        assert!(exif_camera_id(b"definitely not an image").is_none());
        assert!(exif_camera_id(&[]).is_none());
    }

    /// Builds a minimal JPEG (SOI + APP1 EXIF + EOI) carrying BodySerialNumber
    /// (0xA431, ASCII) in the requested byte order. With `in_exif_ifd = true`
    /// the tag lives in a proper Exif SubIFD (what real cameras write, per the
    /// EXIF standard); otherwise it sits directly in IFD0 (tolerated variant).
    /// Hand-rolled because kamadak-exif is read-only.
    fn jpeg_with_exif(serial: &[u8], big_endian: bool, in_exif_ifd: bool) -> Vec<u8> {
        // All multi-byte TIFF values follow the declared byte order.
        let w16 = |v: u16| {
            if big_endian { v.to_be_bytes().to_vec() } else { v.to_le_bytes().to_vec() }
        };
        let w32 = |v: u32| {
            if big_endian { v.to_be_bytes().to_vec() } else { v.to_le_bytes().to_vec() }
        };
        let mut tiff: Vec<u8> = Vec::new();
        if big_endian {
            tiff.extend_from_slice(b"MM\x00\x2a");
        } else {
            tiff.extend_from_slice(b"II\x2a\x00");
        }
        tiff.extend_from_slice(&w32(8)); // offset of IFD0

        let entry = |tag: u16, etype: u16, count: u32, offset: u32| {
            let mut e = Vec::new();
            e.extend_from_slice(&w16(tag));
            e.extend_from_slice(&w16(etype));
            e.extend_from_slice(&w32(count));
            e.extend_from_slice(&w32(offset));
            e
        };

        // Layout: IFD0 @8 (2+12*n+4), [Exif IFD], serial value last.
        let ifd0_end = 8 + 2 + 12 + 4;
        let exif_ifd_off = ifd0_end;
        let exif_ifd_end = exif_ifd_off + if in_exif_ifd { 2 + 12 + 4 } else { 0 };
        let serial_off = exif_ifd_end;

        tiff.extend_from_slice(&w16(1)); // IFD0 entry count
        if in_exif_ifd {
            tiff.extend_from_slice(&entry(0x8769, 4, 1, exif_ifd_off as u32));
        } else {
            tiff.extend_from_slice(&entry(0xa431, 2, serial.len() as u32, serial_off as u32));
        }
        tiff.extend_from_slice(&w32(0)); // no next IFD
        if in_exif_ifd {
            tiff.extend_from_slice(&w16(1)); // Exif IFD entry count
            tiff.extend_from_slice(&entry(0xa431, 2, serial.len() as u32, serial_off as u32));
            tiff.extend_from_slice(&w32(0));
        }
        tiff.extend_from_slice(serial); // value data

        let mut app1: Vec<u8> = b"Exif\x00\x00".to_vec();
        app1.extend_from_slice(&tiff);

        let mut jpeg: Vec<u8> = b"\xff\xd8".to_vec(); // SOI
        jpeg.extend_from_slice(&0xffe1u16.to_be_bytes()); // APP1 marker
        jpeg.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes()); // segment length
        jpeg.extend_from_slice(&app1);
        jpeg.extend_from_slice(b"\xff\xd9"); // EOI
        jpeg
    }

    #[test]
    fn serial_tag_is_extracted_little_and_big_endian() {
        let s = b"1234ABCD\x00";
        assert_eq!(
            exif_camera_id(&jpeg_with_exif(s, false, true)).as_deref(),
            Some("1234ABCD")
        );
        assert_eq!(
            exif_camera_id(&jpeg_with_exif(s, true, true)).as_deref(),
            Some("1234ABCD")
        );
    }

    #[test]
    fn serial_in_ifd0_is_found_by_tag_scan() {
        let s = b"1234ABCD\x00";
        assert_eq!(
            exif_camera_id(&jpeg_with_exif(s, false, false)).as_deref(),
            Some("1234ABCD")
        );
    }

    #[test]
    fn dirty_serial_is_sanitized() {
        let s = b" EOS 5D #42 \x00";
        assert_eq!(
            exif_camera_id(&jpeg_with_exif(s, false, true)).as_deref(),
            Some("EOS_5D__42")
        );
    }
}
