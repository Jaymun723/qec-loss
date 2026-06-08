from qec_loss._fast_qec_loss import LossInstruction


def test_loss_instruction():
    # Test parsing of a valid LOSS instruction
    instr = LossInstruction("LOSS[tag](0.1) 0 1 2 3")
    assert instr.p == 0.1
    assert instr.targets == [0, 1, 2, 3]
    assert instr.tag == "tag"

    instr = LossInstruction("LOSS(0.05) 4 5")
    assert instr.p == 0.05
    assert instr.targets == [4, 5]
    assert instr.tag == ""

    instr = LossInstruction([4, 1], 0.1)
    assert instr.p == 0.1
    assert instr.targets == [4, 1]
    assert instr.tag == ""

    instr = LossInstruction([4, 1], 0.1, "mytag")
    assert instr.p == 0.1
    assert instr.targets == [4, 1]
    assert instr.tag == "mytag"
