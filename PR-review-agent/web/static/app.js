const state = { repositories: [], reviews: [], selectedJobId: null, busy: false };
const byId = (id) => document.getElementById(id);
const dialog = byId("repository-dialog");
let toastTimeout;

function escapeText(value) {
  return String(value ?? "");
}

async function api(path, options = {}) {
  const token = localStorage.getItem("reviewops_dashboard_key");
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 401) {
      localStorage.removeItem("reviewops_dashboard_key");
      showAccessDialog(payload.detail || "Dashboard authentication required");
    }
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

function setServiceStatus(healthy, text) {
  const label = byId("service-status");
  label.textContent = text;
  label.previousElementSibling.style.background = healthy ? "var(--green)" : "var(--red)";
}

function showAccessDialog(message = "") {
  byId("access-error").textContent = message;
  if (!byId("access-dialog").open) byId("access-dialog").showModal();
}

function showToast(message) {
  const toast = byId("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(toastTimeout);
  toastTimeout = window.setTimeout(() => toast.classList.remove("visible"), 3800);
}

function setText(id, value) {
  byId(id).textContent = String(value);
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = escapeText(text);
  return node;
}

function renderStats() {
  const metrics = state.metrics || {};
  const running = Number(metrics.queued || 0) + Number(metrics.running || 0);
  setText("stat-repositories", state.repositories.length);
  setText("stat-pulls", metrics.pull_requests || 0);
  setText("stat-running", running);
  setText("stat-findings", metrics.findings || 0);
}

function renderRepositories() {
  const container = byId("repository-list");
  container.replaceChildren();
  if (!state.repositories.length) {
    const empty = element("div", "empty-state");
    empty.append(element("span", "empty-icon", "⌘"));
    empty.append(element("strong", "", "No repositories connected"));
    empty.append(element("p", "", "Connect a repository your GitHub App can access to start receiving PR events."));
    container.append(empty);
    return;
  }
  for (const repo of state.repositories) {
    const card = element("article", "repo-card");
    const heading = element("div", "repo-card-heading");
    const identity = element("div", "repo-identity");
    const avatar = element("span", "repo-avatar", repo.owner.slice(0, 1).toUpperCase());
    const name = element("div", "repo-name");
    name.append(element("strong", "", `${repo.owner}/${repo.name}`));
    name.append(element("small", "", `Default branch · ${repo.default_branch}`));
    identity.append(avatar, name);
    const link = element("a", "repo-link", "↗");
    link.href = `https://github.com/${encodeURIComponent(repo.owner)}/${encodeURIComponent(repo.name)}`;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.setAttribute("aria-label", `Open ${repo.owner}/${repo.name} on GitHub`);
    heading.append(identity, link);
    card.append(heading);

    const meta = element("div", "repo-meta");
    const prs = element("span");
    prs.append(element("strong", "", repo.pull_requests || 0), document.createTextNode(" pull requests"));
    const reviews = element("span");
    reviews.append(element("strong", "", repo.reviews || 0), document.createTextNode(" reviews"));
    meta.append(prs, reviews);
    card.append(meta);
    card.append(element("div", "repo-index", repo.last_indexed_sha ? `Indexed · ${repo.last_indexed_sha.slice(0, 12)}` : "Index will be built on first review"));

    const form = element("form", "manual-review");
    form.dataset.repositoryId = String(repo.id);
    const input = element("input");
    input.type = "number";
    input.min = "1";
    input.required = true;
    input.placeholder = "PR number";
    input.setAttribute("aria-label", `PR number for ${repo.owner}/${repo.name}`);
    const button = element("button", "button button-secondary", "Run review");
    button.type = "submit";
    form.append(input, button);
    card.append(form);
    container.append(card);
  }
}

function renderReviews() {
  const list = byId("review-list");
  list.replaceChildren();
  if (!state.reviews.length) {
    const empty = element("div", "empty-state");
    empty.append(element("strong", "", "No pull requests yet"));
    empty.append(element("p", "", "New PRs in connected repositories will appear here."));
    list.append(empty);
    renderReviewDetail(null);
    return;
  }
  for (const review of state.reviews) {
    const row = element("button", `review-row${review.id === state.selectedJobId ? " selected" : ""}`);
    row.type = "button";
    row.dataset.jobId = String(review.id);
    row.append(element("span", `status-dot status-${review.status}`));
    const copy = element("span", "review-row-copy");
    const title = review.title || `Pull request #${review.github_pr_number}`;
    copy.append(element("strong", "", `${review.owner}/${review.name} · #${review.github_pr_number} ${title}`));
    const time = review.created_at ? new Date(`${review.created_at.replace(" ", "T")}Z`).toLocaleString() : "";
    copy.append(element("small", "", `${review.head_branch ? `${review.head_branch} · ` : ""}${time}`));
    row.append(copy, element("span", "review-status-label", review.status));
    list.append(row);
  }
  if (!state.reviews.some((item) => item.id === state.selectedJobId)) {
    state.selectedJobId = state.reviews[0].id;
  }
  renderReviewDetail(state.reviews.find((item) => item.id === state.selectedJobId));
}

function renderReviewDetail(payload) {
  const container = byId("review-detail");
  container.replaceChildren();
  if (!payload) {
    const empty = element("div", "empty-detail");
    empty.append(element("span", "empty-icon", "⌁"));
    empty.append(element("h3", "", "Select a review"));
    empty.append(element("p", "", "Review status, evidence, and findings will appear here."));
    container.append(empty);
    return;
  }
  const job = payload.job || payload;
  const findings = payload.findings || [];
  const header = element("div", "detail-header");
  const titleGroup = element("div");
  titleGroup.append(element("div", "eyebrow", `${job.owner}/${job.name} · PR #${job.github_pr_number}`));
  titleGroup.append(element("h3", "", job.title || `Pull request #${job.github_pr_number}`));
  titleGroup.append(element("p", "", `${job.head_branch || "branch unavailable"}${job.author ? ` · opened by ${job.author}` : ""}`));
  header.append(titleGroup, element("span", "status-pill", job.status));
  container.append(header);

  if (job.summary) {
    const summary = element("div", "review-summary");
    summary.textContent = job.summary;
    container.append(summary);
  }
  if (job.error) {
    const error = element("div", "finding-card");
    error.append(element("div", "finding-title", "Review could not be completed"));
    error.append(element("p", "", job.error));
    container.append(error);
  }
  if (payload.metrics) {
    const metrics = payload.metrics;
    const row = element("div", "detail-metrics");
    for (const [label, value] of [
      ["Retrieved chunks", metrics.number_of_chunks ?? 0],
      ["Findings", metrics.number_of_findings ?? findings.length],
      ["Tokens", metrics.tokens_used ?? 0],
    ]) {
      const metric = element("span", "", label);
      metric.append(element("strong", "", value));
      row.append(metric);
    }
    container.append(row);
  }
  if (["queued", "running"].includes(job.status)) {
    const waiting = element("div", "empty-state");
    waiting.append(element("span", "pulse"));
    waiting.append(element("strong", "", job.status === "queued" ? "Review queued" : "Review in progress"));
    waiting.append(element("p", "", "This view refreshes automatically as the worker completes each stage."));
    container.append(waiting);
  } else if (!findings.length && !job.error) {
    const none = element("div", "empty-state");
    none.append(element("span", "empty-icon", "✓"));
    none.append(element("strong", "", "No validated findings"));
    none.append(element("p", "", "The review completed without findings above the configured confidence threshold."));
    container.append(none);
  }
  for (const finding of findings) {
    const card = element("article", "finding-card");
    const findingTitle = element("div", "finding-title");
    findingTitle.append(element("strong", "", finding.title));
    findingTitle.append(element("span", `severity severity-${finding.severity}`, finding.severity));
    card.append(findingTitle);
    card.append(element("div", "finding-location", `${finding.file_path}:${finding.line_start}${finding.line_end !== finding.line_start ? `-${finding.line_end}` : ""} · ${Math.round(finding.confidence * 100)}% confidence`));
    card.append(element("p", "", finding.explanation));
    if (finding.evidence) {
      const evidence = element("div", "finding-evidence");
      evidence.textContent = `Evidence: ${Array.isArray(finding.evidence) ? finding.evidence.join(" · ") : finding.evidence}`;
      card.append(evidence);
    }
    if (finding.suggestion) card.append(element("div", "finding-suggestion", finding.suggestion));
    container.append(card);
  }
}

async function refresh() {
  if (state.busy || document.hidden) return;
  state.busy = true;
  try {
    const [repositories, reviewPage, metrics] = await Promise.all([
      api("/repositories"),
      api("/reviews?limit=100"),
      api("/metrics"),
    ]);
    state.repositories = repositories;
    state.reviews = reviewPage.items || [];
    state.metrics = metrics;
    renderRepositories();
    renderStats();
    const selected = state.reviews.find((item) => item.id === state.selectedJobId) || state.reviews[0];
    state.selectedJobId = selected?.id ?? null;
    renderReviews();
    setServiceStatus(true, "Connected");
  } catch (error) {
    setServiceStatus(false, "API unavailable");
    if (!localStorage.getItem("reviewops_dashboard_key")) {
      showAccessDialog(error.message);
    }
  } finally {
    state.busy = false;
    if (localStorage.getItem("reviewops_dashboard_key")) {
      window.setTimeout(refresh, 5000);
    }
  }
}

byId("dashboard-access").addEventListener("click", () => showAccessDialog());
byId("close-access").addEventListener("click", () => byId("access-dialog").close());
byId("access-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  localStorage.setItem("reviewops_dashboard_key", byId("access-key").value);
  byId("access-key").value = "";
  byId("access-dialog").close();
  await refresh();
});

function openRepositoryDialog() {
  byId("form-error").textContent = "";
  dialog.showModal();
}

byId("add-repository").addEventListener("click", openRepositoryDialog);
byId("add-repository-top").addEventListener("click", openRepositoryDialog);
byId("close-dialog").addEventListener("click", () => dialog.close());
byId("cancel-dialog").addEventListener("click", () => dialog.close());
byId("repository-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = byId("submit-repository");
  button.disabled = true;
  button.textContent = "Connecting…";
  byId("form-error").textContent = "";
  try {
    const full_name = byId("repository-name").value.trim();
    const result = await api("/repositories", {
      method: "POST",
      body: JSON.stringify({ full_name }),
    });
    dialog.close();
    byId("repository-form").reset();
    showToast(`Connected ${result.repository}`);
    await refresh();
  } catch (error) {
    byId("form-error").textContent = error.message;
  } finally {
    button.disabled = false;
    button.textContent = "Connect repository";
  }
});

byId("repository-list").addEventListener("submit", async (event) => {
  if (!event.target.matches(".manual-review")) return;
  event.preventDefault();
  const form = event.target;
  const button = form.querySelector("button");
  button.disabled = true;
  button.textContent = "Queuing…";
  try {
    const result = await api(`/repositories/${form.dataset.repositoryId}/reviews`, {
      method: "POST",
      body: JSON.stringify({ pr_number: Number(form.querySelector("input").value) }),
    });
    state.selectedJobId = result.job_id;
    showToast(result.queued ? "Pull request queued for review" : "This commit already has a review job");
    await refresh();
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Run review";
  }
});

byId("review-list").addEventListener("click", async (event) => {
  const row = event.target.closest("[data-job-id]");
  if (!row) return;
  state.selectedJobId = Number(row.dataset.jobId);
  renderReviews();
  try {
    renderReviewDetail(await api(`/reviews/${state.selectedJobId}`));
  } catch (error) {
    showToast(error.message);
  }
});

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh();
});
refresh();
