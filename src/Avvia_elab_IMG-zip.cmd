::curl -fsS -D - -F "file=@Mio_Test.zip" http://127.0.0.1:8080/anonymize -o Mio_Test_elaborato.zip
curl -fsS -D - -F "file=@WIDER_val.zip" http://127.0.0.1:8080/anonymize -o WIDER_val_elaborato.zip
::curl.exe -fsS -m 3600 -D - -F "file=@$Root\WIDER_val.zip" "http://127.0.0.1:8080/anonymize" -o "$Root\WIDER_val_elaborato.zip"
