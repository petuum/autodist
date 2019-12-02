from autodist.proto import foo_pb2

def test_foo():
    f = foo_pb2.Foo()
    foo_pb2.bar = 'lol'
