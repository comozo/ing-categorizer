// Drives the one-card-per-merchant review flow. Progressive enhancement: every
// card ships with a real <select name="category_{index}"> already carrying the
// AI's suggestion, so the plain form still works with JS disabled (see
// style.css - .native-select is only hidden once .js-active is set below).
// Everything here layers a faster, game-like way to set that same select's
// value, and checkpoints each answer to the server as it's made so a closed
// tab resumes where it left off (see app/sessions.py).
(function () {
  "use strict";

  document.body.classList.add("js-active");

  const deck = document.getElementById("deck");
  if (!deck) return;

  const sessionId = deck.dataset.session;
  const cards = Array.from(deck.querySelectorAll(".card"));
  const playable = cards.filter((c) => c.id !== "summary-card");
  const total = playable.length;

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
    const suggestedChip = grid.querySelector(`.chip[data-category="${cssEscape(grid.dataset.suggested)}"]`);
    if (suggestedChip) {
      suggestedChip.classList.add("chip-suggested");
      suggestedChip.insertAdjacentHTML("beforeend", '<span class="chip-ai-tag">AI pick</span>');
    }
    if (grid.dataset.picked) {
      const pickedChip = grid.querySelector(`.chip[data-category="${cssEscape(grid.dataset.picked)}"]`);
      if (pickedChip) pickedChip.classList.add("chip-picked");
    }
  });

  let index = 0;
  // True during the short pause between an answer and the next card, so a
  // double tap can't answer one card and skip past the next.
  let advancing = false;
  let streak = Number(deck.dataset.streak) || 0;
  let bestStreak = Number(deck.dataset.bestStreak) || 0;
  let xp = Number(deck.dataset.xp) || 0;

  const progressFill = document.getElementById("progress-fill");
  const progressCount = document.getElementById("progress-count");
  const streakCount = document.getElementById("streak-count");
  const streakPill = document.getElementById("streak-pill");
  const xpCount = document.getElementById("xp-count");
  const backBtn = document.getElementById("back-btn");
  const saveState = document.getElementById("save-state");

  function escapeHtml(str) {
    return String(str).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function cssEscape(str) {
    return window.CSS && CSS.escape ? CSS.escape(str) : String(str).replace(/["\\]/g, "\\$&");
  }

  // ---------- checkpointing ----------

  // Answers not yet acknowledged by the server, keyed by merchant index so a
  // changed answer replaces the stale one. Retried until they land, so a flaky
  // connection loses nothing as long as this tab stays open.
  const pending = new Map();
  let inFlight = false;
  let retryTimer = null;

  function checkpoint(answers) {
    answers.forEach((a) => pending.set(a.index, a.category));
    flush();
  }

  function flush() {
    if (inFlight || pending.size === 0) return;
    const batch = Array.from(pending, ([i, category]) => ({ index: i, category }));
    inFlight = true;
    fetch(`/review/${sessionId}/answers`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answers: batch, stats: { xp, streak, best_streak: bestStreak } }),
      keepalive: true,
    })
      .then((resp) => {
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        batch.forEach((a) => {
          if (pending.get(a.index) === a.category) pending.delete(a.index);
        });
        setSaveState(pending.size ? "Saving…" : "Progress saved", false);
      })
      .catch(() => {
        setSaveState("Not saved yet: can't reach the server. Retrying…", true);
        clearTimeout(retryTimer);
        retryTimer = setTimeout(flush, 3000);
      })
      .finally(() => {
        inFlight = false;
        if (pending.size && !retryTimer) flush();
      });
  }

  function setSaveState(text, isError) {
    saveState.textContent = text;
    saveState.classList.toggle("save-error", isError);
    if (!isError) retryTimer = null;
  }

  window.addEventListener("beforeunload", (e) => {
    if (pending.size) e.preventDefault();
  });

  // ---------- game loop ----------

  function showCard(i) {
    cards.forEach((c, idx) => c.classList.toggle("active", idx === i));
    index = i;
    // A chip tapped on the previous card keeps focus once hidden, which would
    // swallow the next Enter (see the keydown handler).
    const focused = document.activeElement;
    if (focused && focused.closest(".card:not(.active)")) focused.blur();
    backBtn.disabled = i === 0;
    if (cards[i].id === "summary-card") {
      document.getElementById("final-xp").textContent = xp;
      document.getElementById("final-streak").textContent = bestStreak;
    }
    updateHud();
  }

  function updateHud() {
    const confirmed = playable.filter((c) => c.dataset.confirmed === "true").length;
    progressFill.style.width = (total ? (confirmed / total) * 100 : 100) + "%";
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

  function setAnswer(card, category) {
    card.querySelector(".native-select").value = category;
    card.querySelectorAll(".chip-picked").forEach((c) => c.classList.remove("chip-picked"));
    const chip = card.querySelector(`.chip[data-category="${cssEscape(category)}"]`);
    if (chip) chip.classList.add("chip-picked");
    card.dataset.confirmed = "true";
  }

  // anchor is what the "+N XP" toast floats up from: the chip itself, or the
  // "Looks right" button when that's what was tapped.
  function pick(card, chip, anchor) {
    setAnswer(card, chip.dataset.category);
    streak += 1;
    bestStreak = Math.max(bestStreak, streak);
    // A merchant covering many transactions is worth more: one tap did more work.
    const base = card.dataset.needsReview === "true" ? 15 : 5;
    const gained = base + Math.min(Number(card.dataset.count) || 1, 10) + Math.min(streak * 2, 20);
    xp += gained;
    flyXP(anchor || chip, gained);
    if (streak % 5 === 0) confettiBurst();
    finishCard(card, chip.dataset.category);
  }

  function skip(card) {
    setAnswer(card, "Uncategorized");
    streak = 0;
    finishCard(card, "Uncategorized");
  }

  function finishCard(card, category) {
    checkpoint([{ index: Number(card.dataset.index), category }]);
    updateHud();
    advancing = true;
    window.setTimeout(() => {
      advancing = false;
      if (index < cards.length - 1) showCard(index + 1);
    }, 420);
  }

  function acceptRest() {
    const remaining = playable.filter((c) => c.dataset.confirmed !== "true");
    const answers = remaining.map((card) => {
      const suggested = card.querySelector(".chip-grid").dataset.suggested;
      setAnswer(card, suggested);
      xp += 5;
      return { index: Number(card.dataset.index), category: suggested };
    });
    checkpoint(answers);
    showCard(cards.length - 1);
  }

  deck.addEventListener("click", (e) => {
    const card = e.target.closest(".card");
    if (!card || card.id === "summary-card" || advancing) return;
    if (e.target.closest("[data-accept-rest]")) return acceptRest();
    if (e.target.closest("[data-skip]")) return skip(card);
    const accept = e.target.closest("[data-accept]");
    if (accept) return pick(card, card.querySelector(".chip-suggested"), accept);
    const chip = e.target.closest(".chip");
    if (chip) pick(card, chip);
  });

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || advancing) return;
    // Let form controls and focused buttons handle their own keys: Enter on a
    // focused chip is already a click.
    if (e.target.closest("select, input, textarea, button, summary")) return;
    const card = cards[index];
    if (e.key === "ArrowLeft" || e.key === "Backspace") {
      if (index > 0) showCard(index - 1);
      e.preventDefault();
      return;
    }
    if (card.id === "summary-card") return;
    if (e.key === "Enter") {
      const chip = card.querySelector(".chip-suggested");
      if (chip) pick(card, chip);
      e.preventDefault();
    } else if (e.key === "s" || e.key === "S") {
      skip(card);
    } else if (e.key === "ArrowRight" && card.dataset.confirmed === "true") {
      showCard(index + 1);
    }
  });

  // Changing a remembered merchant on the summary card is a correction worth
  // keeping too.
  document.querySelectorAll(".remembered-select").forEach((select) => {
    select.addEventListener("change", () => {
      checkpoint([{ index: Number(select.dataset.index), category: select.value }]);
    });
  });

  backBtn.addEventListener("click", () => {
    if (index > 0) showCard(index - 1);
  });

  // Resume at the first merchant not answered yet (the summary card if all are).
  const firstOpen = playable.findIndex((c) => c.dataset.confirmed !== "true");
  showCard(firstOpen === -1 ? cards.length - 1 : firstOpen);
})();
