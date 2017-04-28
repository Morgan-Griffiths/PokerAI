
import ctypes

lib = ctypes.cdll.LoadLibrary("rusteval/target/release/librusteval.dylib")

x = lib.process(b'hi there')

print(x)
