import mlx.core as mx

def fun(a, b, d1, d2):
  x = mx.matmul(a, b, stream=d1)
  for _ in range(500):
      b = mx.exp(b, stream=d2)
  return x, b

a = mx.random.uniform(shape=(4096, 512))
b = mx.random.uniform(shape=(512, 4))

x, b = fun(a, b, mx.gpu, mx.gpu)

print(
    f"x = {x}, b = {b}, "
)