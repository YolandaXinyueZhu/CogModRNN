import jax

devices = jax.devices()
print(devices)

if any(device.device_kind == 'Gpu' for device in devices):
    print("GPU is available and will be used.")
else:
    print("GPU is not available. Falling back to CPU.")

