// Puts a form's earlier values back after the server sends it again.
function restore(values) {
  for (const [name, value] of Object.entries(values)) {
    for (const el of document.getElementsByName(name)) {
      if (el.type === "radio") el.checked = el.value === value;
      else if (el.type === "checkbox") el.checked = Boolean(value);
      else if (el.type !== "file" && el.type !== "submit") el.value = value;
    }
  }
}

document.addEventListener("DOMContentLoaded", () => {
  // The settings switches: a hidden field carries each one's state.
  for (const button of document.querySelectorAll("[role=switch]")) {
    const field = document.getElementsByName(button.dataset.field)[0];
    button.addEventListener("click", () => {
      const on = button.getAttribute("aria-checked") !== "true";
      button.setAttribute("aria-checked", String(on));
      field.value = on ? "on" : "off";
    });
  }
  // An Attach button opens the hidden file field next to it.
  for (const button of document.querySelectorAll("[data-attach]")) {
    const input = document.getElementById(button.dataset.attach);
    button.addEventListener("click", () => input.click());
    input.addEventListener("change", () => {
      document.getElementById(input.id + "_name").textContent = input.files[0] ? input.files[0].name : "No file chosen";
    });
  }
  // The photo picker: pressed tiles go into one hidden field.
  const tiles = [...document.querySelectorAll(".tile")];
  const selected = document.getElementById("selected");
  const update = () => {
    const chosen = tiles.filter((t) => t.getAttribute("aria-pressed") === "true");
    selected.value = chosen.map((t) => t.dataset.tile).join(",");
    document.getElementById("selected_count").textContent = chosen.length;
  };
  for (const tile of tiles) {
    tile.addEventListener("click", () => {
      tile.setAttribute("aria-pressed", String(tile.getAttribute("aria-pressed") !== "true"));
      update();
    });
  }
  const clear = document.querySelector("[data-clear-tiles]");
  if (clear) {
    clear.addEventListener("click", () => {
      tiles.forEach((t) => t.setAttribute("aria-pressed", "false"));
      update();
    });
  }
});
