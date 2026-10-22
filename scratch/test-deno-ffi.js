// scratch/test-deno-ffi.js
const projectRoot = "C:/Users/abdka/Development/foresight-analysis";
const dllPath = `${projectRoot}/dist/sherpa-onnx-c-api.dll`;

try {
  const dylib = Deno.dlopen(dllPath, {
    // Just a basic symbol check
    SherpaOnnxCreateOfflineRecognizer: {
      parameters: ["pointer"],
      result: "pointer",
    },
    SherpaOnnxDestroyOfflineRecognizer: {
      parameters: ["pointer"],
      result: "void",
    }
  });
  console.log("SUCCESS: sherpa-onnx-c-api.dll loaded via Deno FFI!");
  dylib.close();
} catch (e) {
  console.error("FFI Error:", e);
}
