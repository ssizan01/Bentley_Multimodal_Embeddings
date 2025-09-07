"""
NiceGUI app: type a query, see the 5 most relevant images with cosine similarity.

Start:
  python main.py
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any

from nicegui import app, ui, run

import config
from service import search_images_by_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nicegui")

# Serve your local image directory at /static
app.add_static_files("/static", str(config.STATIC_DIR))


async def _do_search(q: str, output_area: ui.element) -> None:
    """Perform the text->image search and render results into output_area."""
    output_area.clear()
    if not q.strip():
        ui.notify("Please enter a query")
        return

    # show a spinner while embedding + querying BigQuery
    with output_area:
        spinner = ui.spinner()
    try:
        # <<< THIS is the key change: run the blocking work off the event loop
        results: List[Dict[str, Any]] = await run.io_bound(
            search_images_by_text, q, config.TOP_K
        )
    except Exception as e:
        with output_area:
            ui.label(f"Search failed: {e.__class__.__name__}: {e}").style("color:#c00;")
        return
    finally:
        spinner.delete()

    with output_area:
        if not results:
            ui.label("No results").style("color:#666;")
            return

        ui.label(f'Top {len(results)} results for “{q}”').style(
            "font-weight:600; font-size:18px; margin:8px 0;"
        )

        row = ui.row().style("gap:16px; flex-wrap: wrap; align-items: stretch;")
        with row:
            for r in results:
                with ui.card().style("padding:10px;"):
                    ui.image(f"/static/{r['rel_path']}").style(
                        "max-width:320px; max-height:240px; object-fit:contain;"
                    )
                    ui.label(r["image_name"]).style("font-size:12px; margin-top:4px;")
                    ui.label(f"cosine similarity: {float(r['cosine_sim']):.4f}").style(
                        "font-size:11px; color:#666;"
                    )


@ui.page("/")  # Root page
def index_page() -> None:
    # centered wrapper that works across NiceGUI versions
    wrapper = ui.column()
    wrapper.style("max-width:1200px; margin:0 auto; padding:16px;")
    with wrapper:
        ui.label("Bentley Multimodal Image Search").style(
            "font-weight:600; font-size:20px; margin-bottom:8px;"
        )
        ui.markdown(
            "Type a description (e.g., *red leather interior*, *Bentley logo on grille*). "
            "We embed your text with Vertex AI Multimodal Embeddings and rank images "
            "in BigQuery by cosine similarity."
        ).style("color:#555; margin-bottom:12px;")

        search_box = ui.input(
            label="Search images",
            placeholder='e.g., "red leather interior", "chrome grille", "steering wheel logo"',
        ).props("clearable")
        search_box.style("width:100%; max-width:720px;")

        # Results container
        output = ui.column().style("margin-top:16px;")

        # Define async handlers (no ui.run_async)
        async def trigger_search() -> None:
            await _do_search(search_box.value, output)

        async def on_enter(_e) -> None:
            await trigger_search()

        # Controls
        controls = ui.row().style("gap:8px; align-items:center; margin-top:8px;")
        with controls:
            ui.button("Search", on_click=trigger_search)
            ui.button("Clear", on_click=lambda: (search_box.set_value(""), output.clear()))

        # Enter-to-search
        search_box.on("keydown.enter", on_enter)

        # Helpful hint
        with ui.expansion("Troubleshooting images"):
            ui.label(
                "Make sure your files are under "
                f"{config.STATIC_DIR} and accessible via /static/<rel_path>.\n"
                "Example: if a row has rel_path = photos/car.jpg, "
                "the app loads /static/photos/car.jpg"
            ).style("font-size:12px; color:#555;")


# No __main__ guard (per your preference)
ui.run(title="Bentley Multimodal Image Search")
