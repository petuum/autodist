from autodist.strategy.base import Strategy


def test_expr():
    s = Strategy()
    print(s.id)


def test_serialization():
    s = Strategy()
    s.serialize()
    s2 = Strategy.deserialize(s.id)
    print(s, s2)
    assert s.id == s2.id
    assert s.path == s2.path
    assert s.node_config == s2.node_config
    assert s.graph_config == s2.graph_config
