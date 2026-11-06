import pytest

from provenance import corpus


@pytest.fixture(scope="module")
def df():
    return corpus.load()


def test_shape(df):
    assert len(df) == 8381
    assert df.source_id.nunique() == 493
    assert set(df.groupby("source_id").size().unique()) == {17}


def test_source_annotations_survive_editing(df):
    assert df.groupby("source_id").subfield.nunique().max() == 1
    source = df[df.operation == "original"].set_index("source_id")
    expected = (df.normalized_edit_distance * df.source_id.map(source.char_count)).round()
    assert (df.edit_chars == expected).all()


def test_effective_presence_excludes_all_unchanged_edits(df):
    selected = df[corpus.effective_presence_mask(df)]
    assert len(selected[selected.operation == "original"]) == 493
    assert not ((selected.intervention_present == 1) & (selected.identical == 1)).any()
    assert len(selected[selected.group_split == "test"]) == 1332


def test_one_record_per_cell(df):
    cells = df.groupby(["source_id", "operation", "provider"]).size()
    assert cells.max() == 1
    assert len(cells) == 8381


def test_split_is_document_level(df):
    assert df.groupby("source_id").group_split.nunique().max() == 1
    assert df.groupby("group_id").source_id.nunique().max() == 1


def test_edited_input_matches_source_output(df):
    orig = df[df.operation == "original"].set_index("source_id").output_sha256
    ed = df[df.operation != "original"]
    assert (ed.input_sha256.values == ed.source_id.map(orig).values).all()


def test_levels_and_labels(df):
    assert set(df.intervention_level.unique()) == {0, 1, 2, 3, 4}
    assert df[df.operation == "original"].intervention_present.sum() == 0
    assert df[df.intervention_present == 1].operation.nunique() == 4


def test_missing_text_fields_absent(df):
    for col in ["abstract_clean", "char_count", "normalized_edit_distance"]:
        assert col in df.columns
    assert (df.char_count > 0).all()
    assert (df.normalized_edit_distance >= 0).all()
