"""The two collections of the publication: found by name, filled without duplicates."""

from typing import Any

from driftmap.core.hub import Hub
from driftmap.core.schemas import DriftmapError, ExitCode
from driftmap.core.settings import Settings

SHELVES = {"models": "models_collection", "data": "data_collection"}


def find(settings: Settings, hub: Hub) -> dict[str, Any | None]:
    """The owner's models and data collections with all their items; None for one that does not exist."""
    owned = hub.list_collections(settings.owner)
    found = {}
    for shelf, field in SHELVES.items():
        name = getattr(settings.repos, field)
        # a slug is the title as first given plus an id, so a retitled collection still matches
        matches = [c for c in owned if c.title == name or c.slug.rpartition("-")[0] == f"{settings.owner}/{name}"]
        if len(matches) > 1:
            raise DriftmapError(ExitCode.INTEGRITY, f"{len(matches)} collections match {name}; keep one")
        found[shelf] = hub.get_collection(matches[0].slug) if matches else None
    return found


def holds(collection: Any, repo: str, item_type: str) -> bool:
    return any(item.item_id == repo and item.item_type == item_type for item in collection.items)


def add(hub: Hub, collection: Any, repo: str, item_type: str) -> bool:
    """List a repo in a collection; False, and no call, when it is listed already."""
    if holds(collection, repo, item_type):
        return False
    hub.add_collection_item(collection.slug, repo, item_type)
    return True
