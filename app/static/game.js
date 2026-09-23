// Drives the one-card-at-a-time review flow. Progressive enhancement: every
// card ships with a real <select name="category_{id}"> already carrying the
// AI's suggestion, so the plain multi-row form still works with JS disabled
// (see style.css - .native-select is only hidden once .js-active is set
// below). Everything here does is layer a faster, game-like way to set that
// same select's value.
(function () {
  "use strict";

  document.body.classList.add("js-active");

  const deck = document.getElementById("deck");
  if (!deck) return;

  const cards = Array.from(deck.querySelectorAll(".card"));
  const total = cards.length - 1; // exclude the summary card
  if (total <= 0) return;

  const groups = JSON.parse(document.getElementById("category-groups-data").textContent);

  const chipsHTML = groups
    .map((group) =>
      group.items
        .map(
          (item) =>
            `<button type="button" class="chip" data-category="${escapeHtml(item.name)}" style="--chip-color:${escapeHtml(group.color)}">` +
            `<span class="chip-icon">${item.icon}</span><span class="chip-label">${escapeHtml(item.name)}</span>` +
            `</button>`
        )
        .join("")
    )
    .join("");

  deck.querySelectorAll(".chip-grid").forEach((grid) => {
    grid.innerHTML = chipsHTML;
    const suggested = grid.dataset.suggested;
    const suggestedChip = grid.querySelector(`.chip[data-category="${cssEscape(suggested)}"]`);
    if (suggestedChip) {
      suggestedChip.classList.add("chip-suggested");
      suggestedChip.insertAdjacentHTML("beforeend", '<span class="chip-ai-tag">AI pick</span>');
    }
  });

  let index = 0;
  let streak = 0;
  let bestStreak = 0;
  let xp = 0;

  const progressFill = document.getElementById("progress-fill");
  const progressCount = document.getElementById("progress-count");
  const streakCount = document.getElementById("streak-count");
  const streakPill = document.getElementById("streak-pill");
  const xpCount = document.getElementById("xp-count");
  const backBtn = document.getElementById("back-btn");

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function cssEscape(str) {
    return window.CSS && CSS.escape ? CSS.escape(str) : String(str).replace(/["\\]/g, "\\$&");
  }

  function showCard(i) {
    cards.forEach((c, idx) => c.classList.toggle("active", idx === i));
    index = i;
    backBtn.disabled = i === 0;
    if (cards[i].id === "summary-card") {
      document.getElementById("final-xp").textContent = xp;
      document.getElementById("final-streak").textContent = bestStreak;
    }
    updateHud();
  }

  function updateHud() {
    const confirmed = cards.filter((c) => c.dataset.confirmed === "true").length;
    progressFill.style.width = (confirmed / total) * 100 + "%";
    progressCount.textContent = confirmed;
    streakCount.textContent = streak;
    xpCount.textContent = xp;
    streakPill.classList.toggle("streak-hot", streak >= 3);
  }

  function flyXP(anchor, amount) {
    const rect = anchor.getBoundingClientRect();
    const el = document.createElement("div");
    el.className = "xp-toast";
    el.textContent = "+" + amount + " XP";
    el.style.left = rect.left + rect.width / 2 + "px";
    el.style.top = rect.top + "px";
    document.body.appendChild(el);
    el.addEventListener("animationend", () => el.remove());
  }

  function confettiBurst() {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const colors = groups.map((g) => g.color);
    for (let i = 0; i < 18; i++) {
      const piece = document.createElement("div");
      piece.className = "confetti-piece";
      piece.style.left = 50 + (Math.random() * 40 - 20) + "vw";
      piece.style.background = colors[i % colors.length];
      piece.style.animationDelay = Math.random() * 0.15 + "s";
      piece.style.setProperty("--drift", Math.random() * 120 - 60 + "px");
      document.body.appendChild(piece);
      piece.addEventListener("animationend", () => piece.remove());
    }
  }

  function advance() {
    if (index < cards.length - 1) showCard(index + 1);
  }

  deck.addEventListener("click", (e) => {
    const skipBtn = e.target.closest("[data-skip]");
    const chip = e.target.closest(".chip");
    if (!skipBtn && !chip) return;

    const card = e.target.closest(".card");
    if (!card || card.id === "summary-card") return;

    const select = card.querySelector(".native-select");

    if (skipBtn) {
      select.value = "Uncategorized";
      streak = 0;
    } else {
      select.value = chip.dataset.category;
      card.querySelectorAll(".chip-picked").forEach((c) => c.classList.remove("chip-picked"));
      chip.classList.add("chip-picked");
      streak += 1;
      bestStreak = Math.max(bestStreak, streak);
      const base = card.dataset.needsReview === "true" ? 15 : 5;
      const bonus = Math.min(streak * 2, 20);
      const gained = base + bonus;
      xp += gained;
      flyXP(chip, gained);
      if (streak > 0 && streak % 5 === 0) confettiBurst();
    }

    select.dispatchEvent(new Event("change"));
    card.dataset.confirmed = "true";
    updateHud();
    window.setTimeout(advance, 420);
  });

  backBtn.addEventListener("click", () => {
    if (index > 0) showCard(index - 1);
  });

  showCard(0);
})();
