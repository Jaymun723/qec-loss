import qec_loss


def test_public_api_exports():
    expected = set(qec_loss.__all__)
    assert expected == {
        "__version__",
        "LossInstruction",
        "LossyCircuit",
        "ForwardSampler",
        "SampleBatch",
        "Sampler",
        "DetsRerouter",
        "PauliRerouter",
        "get_stabilizers",
        "MonakaBuilder",
        "LifeCycleManager",
        "LifeSegment",
        "get_loss_dem",
        "get_loss_segment_dem",
        "get_loss_segment_dems",
        "combine_circuits_into_dem",
        "F2Tensor",
        "PackedF2Matrix",
        "add_loss_noise",
        "PhysicalFrameDecoder",
        "canonicalize_dem_components",
        "decompose_errors",
        "is_graphlike",
        "combine_dems",
    }
    for name in expected:
        assert hasattr(qec_loss, name)


def test_submodule_reexports():
    from qec_loss.circuit import LossInstruction, LossyCircuit
    from qec_loss.f2_tensor import F2Tensor, PackedF2Matrix
    from qec_loss.monaka import MonakaBuilder, get_loss_dem, get_loss_segment_dem, get_loss_segment_dems
    from qec_loss.observable import DetsRerouter, PauliRerouter
    from qec_loss.sampler import ForwardSampler, SampleBatch

    assert LossInstruction is qec_loss.LossInstruction
    assert LossyCircuit is qec_loss.LossyCircuit
    assert ForwardSampler is qec_loss.ForwardSampler
    assert SampleBatch is qec_loss.SampleBatch
    assert DetsRerouter is qec_loss.DetsRerouter
    assert PauliRerouter is qec_loss.PauliRerouter
    assert MonakaBuilder is qec_loss.MonakaBuilder
    assert get_loss_dem is qec_loss.get_loss_dem
    assert get_loss_segment_dem is qec_loss.get_loss_segment_dem
    assert get_loss_segment_dems is qec_loss.get_loss_segment_dems
    assert F2Tensor is qec_loss.F2Tensor
    assert PackedF2Matrix is qec_loss.PackedF2Matrix
