import pytest

from agent.stores.sqlite_store import SqliteSessionRepository


@pytest.mark.asyncio
async def test_sqlite_store_crud_and_slots(tmp_path):
    db_path = str(tmp_path / "el_sessions.db")
    repo = SqliteSessionRepository(db_path=db_path)
    await repo.init()

    # create a session
    sid = await repo.create_session(
        user_id="u1",
        topic="postmortem: payment outage",
        goal_kind="postmortem",
        created_at_iso="2025-11-21T12:00:00Z",
        thread_id=10101,
        language="Japanese",
    )
    assert isinstance(sid, int) and sid > 0

    # add messages (assistant then user)
    await repo.add_message(session_id=sid, ts_iso="2025-11-21T12:00:10Z", role="assistant", text="最初の質問")
    await repo.add_message(session_id=sid, ts_iso="2025-11-21T12:00:20Z", role="user", text="最初の回答")

    # upsert slot metadata, then set value
    await repo.upsert_slot(
        session_id=sid,
        slot={
            "name": "summary",
            "description": "incident summary",
            "type": "postmortem_summary",
            "importance": 1.0,
            "filled_ratio": 0.0,
        },
    )
    await repo.set_slot_value(session_id=sid, slot_name="summary", value="14:30から2時間停止", source_kind="user", filled_ratio=1.0, last_filled_ts=1732180000.0)

    # get by thread
    rec = await repo.get_session_by_thread(user_id="u1", thread_id=10101)
    assert rec and rec.id == sid and rec.goal_kind == "postmortem" and rec.language == "Japanese"

    # slots list reflects value and ratio
    slots = await repo.get_slots(session_id=sid)
    s = next((x for x in slots if x.get("name") == "summary"), None)
    assert s is not None
    assert s.get("value", "").startswith("14:30")
    assert float(s.get("filled_ratio") or 0.0) >= 1.0

    # iter messages are ordered
    msgs = [m async for m in repo.iter_messages(session_id=sid)]
    assert len(msgs) == 2
    assert msgs[0][1] == "assistant" and "最初の質問" in msgs[0][2]
    assert msgs[1][1] == "user" and "最初の回答" in msgs[1][2]


