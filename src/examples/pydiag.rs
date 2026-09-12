// Diagnostica PyO3: replica l'ordine di import di retrain.py (torch prima,
// poi onnx/onnxscript) e prova la cure add_dll_directory per onnx.
use pyo3::prelude::*;

fn try_import(py: Python, name: &str) {
    match py.import(name) {
        Ok(m) => {
            let v: String = m
                .getattr("__version__")
                .and_then(|a| a.extract())
                .unwrap_or_else(|_| "?".to_string());
            println!("{name} OK {v}");
        }
        Err(e) => {
            println!("{name} FAIL:");
            e.print_and_set_sys_last_vars(py);
        }
    }
}

const CURE: &str = r#"
import os, sys
sp = [p for p in sys.path if 'site-packages' in p][0]
for sub in ('onnx', 'onnxscript', 'onnx_ir', 'onnxruntime', 'google/protobuf'):
    d = os.path.join(sp, *sub.split('/'))
    if os.path.isdir(d):
        os.add_dll_directory(d)
        print('dll-dir registered:', d)
"#;

fn main() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        try_import(py, "torch");
        match py.run(CURE, None, None) {
            Ok(_) => println!("cure applied"),
            Err(e) => {
                println!("cure FAILED:");
                e.print_and_set_sys_last_vars(py);
            }
        }
        try_import(py, "onnx");
        try_import(py, "onnxscript");
    });
}
