"""Offline tests for ``GET /v1/providers``.

The app runs with fakes for the store and the models; see ``conftest.py``.
"""

from __future__ import annotations


def test_providers_lists_the_catalogue(client):
    """Every catalogue provider and model, with what a client needs to pick one."""
    from config.providers import ALL_PROVIDERS

    body = client.get("/v1/providers").json()
    assert body["total"] == len(ALL_PROVIDERS)
    assert [p["id"] for p in body["items"]] == [p.id for p in ALL_PROVIDERS]
    gemini = body["items"][0]
    lite = next(m for m in gemini["models"] if m["id"] == "gemini-3.5-flash-lite")
    assert lite == {
        "id": "gemini-3.5-flash-lite",
        "display_name": "Gemini 3.5 Flash-Lite",
        "supports_vision": True,
        "accepts_temperature": False,
        "default_max_tokens": 512,
        "input_price_per_1m": 0.30,
        "output_price_per_1m": 2.50,
    }

def test_providers_hides_server_internals(client):
    for provider in client.get("/v1/providers").json()["items"]:
        assert set(provider) == {"id", "display_name", "models"}

def test_every_listed_model_is_accepted_by_query(client, backend):
    """The list is the same catalogue the query route checks, so nothing listed is a 422."""
    for provider in client.get("/v1/providers").json()["items"]:
        for model in provider["models"]:
            llm = {"provider": provider["id"], "model": model["id"], "api_key": "k"}
            response = client.post("/v1/query", json={"query": "x", "llm": llm})
            assert response.status_code == 200, (provider["id"], model["id"], response.text)

def test_providers_answers_without_the_store(client, backend):
    backend.store_error = "disk gone"
    assert client.get("/v1/providers").status_code == 200
