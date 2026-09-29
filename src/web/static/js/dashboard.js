/**
 * LinkedIn & Buffer Content Automation Studio - Client Controller
 * Manages Creative Studio, Client Hub, Admin Center, and Real-time Mockups.
 */

// Global State
const appState = {
  activeRole: 'creative',
  config: null,
  pipelineStatus: null,
  currentPost: null,
  isGenerating: false,
  isPublishing: false,
};

// DOM Initialization
document.addEventListener('DOMContentLoaded', async () => {
  setupRoleSwitcher();
  setupEventListeners();
  await loadAppConfig();
  await loadStandbyPhotoStatus();
  await refreshPipelineStatus();
  await loadBufferProfiles();
  await checkCloudinary();
});


// -----------------------------------------------------------------------------
// Role Navigation & Switching
// -----------------------------------------------------------------------------
function setupRoleSwitcher() {
  const tabButtons = document.querySelectorAll('.role-tab-btn');
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const role = btn.dataset.role;
      switchRole(role);
    });
  });
}

function switchRole(roleName) {
  appState.activeRole = roleName;

  // Update button active state
  document.querySelectorAll('.role-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.role === roleName);
  });

  // Switch workspace visibility
  document.querySelectorAll('.workspace-view').forEach(view => {
    view.classList.toggle('active', view.id === `view-${roleName}`);
  });

  // Role-specific refreshes
  if (roleName === 'admin') {
    refreshAdminData();
  } else if (roleName === 'client') {
    refreshClientData();
  }
}

// -----------------------------------------------------------------------------
// Configuration & Data Fetching
// -----------------------------------------------------------------------------
async function loadAppConfig() {
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    appState.config = data;
    populateFormSelectors(data);
    renderPillarsInClientHub(data.persona?.content_pillars || []);
  } catch (err) {
    showToast('Failed to load configuration: ' + err.message, 'error');
  }
}

async function refreshPipelineStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    appState.pipelineStatus = data;

    updateHeaderBadge(data);
    renderAdminCheckpoint(data.state);
    renderRecentHistory(data.recent_history || []);

    // If there is an active pending post and no post in editor, load it into creative preview
    if (data.has_pending && data.state?.pending_post && !appState.currentPost) {
      const pending = data.state.pending_post;
      setPostContent(pending.content, pending.metadata, pending.cycle_id, pending.image_data);
    }
  } catch (err) {
    console.error('Error fetching pipeline status:', err);
  }
}


async function loadBufferProfiles() {
  try {
    const res = await fetch('/api/profiles');
    const data = await res.json();
    if (data.success) {
      renderBufferChannels(data.profiles);
    }
  } catch (err) {
    console.error('Error fetching Buffer profiles:', err);
  }
}

function updateHeaderBadge(statusData) {
  const dot = document.getElementById('pipeline-status-dot');
  const text = document.getElementById('pipeline-status-text');
  if (!dot || !text) return;

  const status = statusData.state?.status || 'IDLE';
  dot.className = 'status-dot';

  if (status === 'IDLE') {
    dot.classList.add('idle');
    text.textContent = 'System Ready';
  } else if (status === 'GENERATED' || status === 'FAILED') {
    dot.classList.add('pending');
    text.textContent = `Draft Checkpoint (${statusData.state?.cycle_id || 'Active'})`;
  }
}

// -----------------------------------------------------------------------------
// Form & Prompt Builder
// -----------------------------------------------------------------------------
function populateFormSelectors(config) {
  const pillarSelect = document.getElementById('select-pillar');
  const hookSelect = document.getElementById('select-hook');
  const providerSelect = document.getElementById('select-provider');

  if (pillarSelect && config.persona?.content_pillars) {
    pillarSelect.innerHTML = '<option value="">Auto-Select (Dynamic Rotation)</option>';
    config.persona.content_pillars.forEach(p => {
      pillarSelect.innerHTML += `<option value="${p.id}">${p.name}</option>`;
    });
  }

  if (hookSelect && config.prompts?.hook_styles) {
    hookSelect.innerHTML = '<option value="">Auto-Select (Smart Strategy)</option>';
    Object.entries(config.prompts.hook_styles).forEach(([k, v]) => {
      hookSelect.innerHTML += `<option value="${k}">${v.name}</option>`;
    });
  }

  if (providerSelect && config.llm?.active_provider) {
    providerSelect.value = config.llm.active_provider;
  }
}

function insertIdea(text) {
  const mindInput = document.getElementById('input-mind');
  if (mindInput) {
    mindInput.value = text;
    mindInput.focus();
  }
}

// -----------------------------------------------------------------------------
// Content Generation & Live Preview
// -----------------------------------------------------------------------------
function setupEventListeners() {
  // Generate Form
  const genForm = document.getElementById('form-generate');
  if (genForm) {
    genForm.addEventListener('submit', handleGenerateSubmit);
  }

  // Standby photo upload listener
  document.getElementById('input-standby-upload')?.addEventListener('change', handleStandbyPhotoUpload);

  // Regenerate Image Only listener
  document.getElementById('btn-regenerate-image')?.addEventListener('click', handleRegenerateImageOnly);

  // Live Editor text sync with LinkedIn feed mockup
  const editor = document.getElementById('live-post-editor');
  if (editor) {
    editor.addEventListener('input', () => {
      const text = editor.value;
      updateMockupBody(text);
      updateTelemetry(text);
    });
  }

  // Quick action buttons
  document.getElementById('btn-publish-draft')?.addEventListener('click', handlePublishDraft);
  document.getElementById('btn-save-checkpoint')?.addEventListener('click', handleSaveCheckpoint);
  document.getElementById('btn-regenerate')?.addEventListener('click', handleRegenerate);
  document.getElementById('btn-copy-post')?.addEventListener('click', handleCopyPost);
  document.getElementById('btn-clear-state')?.addEventListener('click', handleClearState);
}

async function handleGenerateSubmit(e) {
  if (e) e.preventDefault();
  if (appState.isGenerating) return;

  const btn = document.getElementById('btn-generate-submit');
  const userMind = document.getElementById('input-mind')?.value.trim();
  const pillarId = document.getElementById('select-pillar')?.value;
  const hookStyle = document.getElementById('select-hook')?.value;
  const provider = document.getElementById('select-provider')?.value;
  const generateImage = document.getElementById('toggle-generate-image')?.checked ?? true;

  setGeneratingState(true);

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_mind: userMind || null,
        pillar_id: pillarId || null,
        hook_style: hookStyle || null,
        provider: provider || null,
        generate_image: generateImage
      })
    });

    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || 'Generation failed');
    }

    setPostContent(data.content, data.metadata, data.cycle_id, data.image_data);
    showToast('✨ LinkedIn post & Qwen-Image generated successfully!', 'success');
    refreshPipelineStatus();
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    setGeneratingState(false);
  }
}

function setGeneratingState(isGenerating) {
  appState.isGenerating = isGenerating;
  const btn = document.getElementById('btn-generate-submit');
  const loader = document.getElementById('mockup-loading-overlay');
  if (btn) {
    btn.disabled = isGenerating;
    btn.innerHTML = isGenerating ? '<div class="spinner"></div> Synthesizing Post & Image...' : '⚡ Generate LinkedIn Post';
  }
  if (loader) {
    loader.style.display = isGenerating ? 'flex' : 'none';
  }
}

function setPostContent(content, metadata = {}, cycleId = null, imageData = null) {
  appState.currentPost = { content, metadata, cycleId, imageData };

  const editor = document.getElementById('live-post-editor');
  if (editor) {
    editor.value = content;
  }

  updateMockupBody(content);
  updateTelemetry(content);
  updateMockupImage(imageData);

  const cycleBadge = document.getElementById('post-cycle-badge');
  if (cycleBadge && cycleId) {
    cycleBadge.textContent = `Cycle #${cycleId}`;
  }
}

function updateMockupImage(imageData) {
  const imgContainer = document.getElementById('linkedin-mockup-image-container');
  const imgEl = document.getElementById('linkedin-mockup-image');
  const promptEditor = document.getElementById('live-image-prompt-editor');
  const cdnBadge = document.getElementById('mockup-cloudinary-badge');

  if (imageData && (imageData.preview_url || imageData.image_url)) {
    if (imgEl) {
      imgEl.src = imageData.preview_url || imageData.image_url;
    }
    if (imgContainer) {
      imgContainer.style.display = 'block';
    }
    if (promptEditor && imageData.prompt) {
      promptEditor.value = imageData.prompt;
    }
    if (cdnBadge) {
      if (imageData.cloudinary_url) {
        cdnBadge.style.display = 'inline';
        cdnBadge.title = `Cloudinary CDN: ${imageData.cloudinary_url}`;
      } else {
        cdnBadge.style.display = 'none';
      }
    }
  } else {
    if (imgContainer) {
      imgContainer.style.display = 'none';
    }
    if (promptEditor) {
      promptEditor.value = '';
    }
    if (cdnBadge) {
      cdnBadge.style.display = 'none';
    }
  }
}

async function handleRegenerateImageOnly() {
  const promptEditor = document.getElementById('live-image-prompt-editor');
  const customPrompt = promptEditor ? promptEditor.value.trim() : '';
  const editor = document.getElementById('live-post-editor');
  const postContent = editor ? editor.value.trim() : (appState.currentPost?.content || '');

  const btn = document.getElementById('btn-regenerate-image');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Synthesizing...';
  }

  try {
    const res = await fetch('/api/generate-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt: customPrompt || null,
        post_content: postContent,
        cycle_id: appState.currentPost?.cycleId || null,
        topic: appState.currentPost?.metadata?.topic || null,
        pillar_name: appState.currentPost?.metadata?.pillar_name || null
      })
    });

    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || 'Image generation failed');
    }

    if (appState.currentPost) {
      appState.currentPost.imageData = data.image_data;
    }
    updateMockupImage(data.image_data);
    showToast('🎨 Qwen-Image-3.0 visual updated successfully!', 'success');
  } catch (err) {
    showToast('Image generation error: ' + err.message, 'error');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '🔄 Regenerate Image Only';
    }
  }
}

async function loadStandbyPhotoStatus() {
  try {
    const res = await fetch('/api/standby-photo');
    const data = await res.json();
    const statusEl = document.getElementById('standby-photo-status');
    const thumbEl = document.getElementById('standby-photo-thumb');
    const badgeEl = document.getElementById('mockup-standby-badge');

    if (data.exists) {
      if (statusEl) statusEl.textContent = `📸 Standby Photo: Ready (${data.filename})`;
      if (thumbEl) {
        thumbEl.src = data.preview_url + '?t=' + Date.now();
        thumbEl.style.display = 'block';
      }
      if (badgeEl) badgeEl.textContent = `📸 Standby Reference Active`;
    } else {
      if (statusEl) statusEl.textContent = '⚠️ Standby Photo Missing';
      if (thumbEl) thumbEl.style.display = 'none';
      if (badgeEl) badgeEl.textContent = 'Text-to-Image Fallback';
    }
  } catch (err) {
    console.error('Error fetching standby photo status:', err);
  }
}

async function handleStandbyPhotoUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  try {
    showToast('Uploading personal standby photo for Qwen-Image-3.0...', 'info');
    const res = await fetch('/api/standby-photo', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || 'Upload failed');
    }
    showToast('📸 Standby photo uploaded and saved!', 'success');
    await loadStandbyPhotoStatus();
  } catch (err) {
    showToast('Failed to upload standby photo: ' + err.message, 'error');
  }
}


function updateMockupBody(text) {
  const mockupBody = document.getElementById('linkedin-mockup-body');
  if (!mockupBody) return;

  if (!text.trim()) {
    mockupBody.innerHTML = '<span style="color: #94a3b8; font-style: italic;">Your generated LinkedIn draft will appear here in real-time...</span>';
    return;
  }

  // Format hashtags with LinkedIn blue
  let formatted = text.replace(/#([\w\u0590-\u05ff]+)/g, '<span class="hashtag">#$1</span>');
  mockupBody.innerHTML = formatted;
}

function updateTelemetry(text) {
  const charCountEl = document.getElementById('telemetry-char-count');
  const readTimeEl = document.getElementById('telemetry-read-time');
  const hookStatusEl = document.getElementById('telemetry-hook-status');

  const len = text.length;
  if (charCountEl) {
    charCountEl.textContent = `${len} / 3000`;
    charCountEl.style.color = len > 3000 ? 'var(--color-danger)' : 'var(--text-primary)';
  }

  if (readTimeEl) {
    const words = text.split(/\s+/).filter(Boolean).length;
    const mins = Math.max(1, Math.round(words / 200));
    readTimeEl.textContent = `~${mins} min read (${words} words)`;
  }

  if (hookStatusEl) {
    const lines = text.split('\n').filter(l => l.trim());
    if (lines.length > 0 && lines[0].length < 140) {
      hookStatusEl.innerHTML = '<span style="color: var(--color-success)">✓ Punchy Hook</span>';
    } else {
      hookStatusEl.innerHTML = '<span style="color: var(--color-warning)">⚠ Check Hook Length</span>';
    }
  }
}

// -----------------------------------------------------------------------------
// Human-in-the-Loop Actions (Publish, Checkpoint, Regenerate)
// -----------------------------------------------------------------------------
async function handlePublishDraft() {
  const editor = document.getElementById('live-post-editor');
  const text = editor ? editor.value.trim() : '';

  if (!text) {
    showToast('Cannot publish empty post.', 'error');
    return;
  }

  const dryRunCheckbox = document.getElementById('toggle-dry-run');
  const isDryRun = dryRunCheckbox ? dryRunCheckbox.checked : false;

  const btn = document.getElementById('btn-publish-draft');
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Sending Draft...';
  }

  try {
    const res = await fetch('/api/publish-draft', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: text,
        dry_run: isDryRun,
        image_url: appState.currentPost?.imageData?.image_url || null,
        metadata: appState.currentPost?.metadata
      })
    });

    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || 'Failed to publish draft');
    }

    const modeMsg = isDryRun ? '[Dry-Run Simulation]' : '[Live Buffer Draft]';
    const imgNote = appState.currentPost?.imageData ? ' with Qwen visual asset' : '';
    showToast(`🚀 Successfully published draft to Buffer! ${modeMsg}${imgNote}`, 'success');
    refreshPipelineStatus();
  } catch (err) {
    showToast('Buffer error: ' + err.message, 'error');
    refreshPipelineStatus();
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = '🚀 Save Draft to Buffer';
    }
  }
}

async function handleSaveCheckpoint() {
  const editor = document.getElementById('live-post-editor');
  const text = editor ? editor.value.trim() : '';

  if (!text) return;

  try {
    const res = await fetch('/api/save-checkpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: text,
        metadata: appState.currentPost?.metadata,
        image_data: appState.currentPost?.imageData || null
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`💾 Saved post & visual to state checkpoint (#${data.cycle_id})`, 'info');
      refreshPipelineStatus();
    }
  } catch (err) {
    showToast('Save failed: ' + err.message, 'error');
  }
}


function handleRegenerate() {
  handleGenerateSubmit();
}

function handleCopyPost() {
  const editor = document.getElementById('live-post-editor');
  const text = editor ? editor.value : '';
  if (!text) return;

  navigator.clipboard.writeText(text).then(() => {
    showToast('📋 Copied post text to clipboard!', 'info');
  });
}

async function handleClearState() {
  try {
    await fetch('/api/clear-state', { method: 'POST' });
    showToast('State checkpoint cleared.', 'info');
    refreshPipelineStatus();
  } catch (err) {
    showToast('Failed to clear state', 'error');
  }
}

// -----------------------------------------------------------------------------
// Client Hub Rendering
// -----------------------------------------------------------------------------
function renderPillarsInClientHub(pillars) {
  const grid = document.getElementById('client-pillars-grid');
  if (!grid) return;

  grid.innerHTML = '';
  pillars.forEach(p => {
    grid.innerHTML += `
      <div class="pillar-card">
        <h4 style="font-size: 15px; font-weight: 700; color: var(--text-primary); margin-bottom: 6px;">${p.name}</h4>
        <p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 12px;">${p.description}</p>
        <div style="font-size: 12px; color: var(--color-accent); font-weight: 600;">
          💡 ${p.sample_topics ? p.sample_topics.length : 0} Core Strategy Topics
        </div>
      </div>
    `;
  });
}

function refreshClientData() {
  refreshPipelineStatus();
}

// -----------------------------------------------------------------------------
// Admin Center Rendering & Tests
// -----------------------------------------------------------------------------
function renderBufferChannels(channels) {
  const container = document.getElementById('admin-channels-list');
  if (!container) return;

  container.innerHTML = '';
  channels.forEach(ch => {
    const isLi = ch.is_linkedin;
    container.innerHTML += `
      <div style="display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; background: var(--bg-surface-elevated); border-radius: var(--radius-md); margin-bottom: 8px; border: 1px solid var(--border-subtle);">
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="width: 32px; height: 32px; border-radius: 50%; background: ${isLi ? '#0a66c2' : '#1da1f2'}; display: flex; align-items: center; justify-content: center; color: #fff; font-weight: 700; font-size: 12px;">
            ${isLi ? 'in' : '𝕏'}
          </div>
          <div>
            <div style="font-weight: 700; font-size: 13px;">${ch.display_name}</div>
            <div style="font-size: 12px; color: var(--text-secondary);">${ch.formatted_service} • ID: <code>${ch.id}</code></div>
          </div>
        </div>
        <span style="font-size: 11px; font-weight: 700; padding: 4px 8px; border-radius: 9999px; background: rgba(16, 185, 129, 0.15); color: var(--color-success);">
          Active
        </span>
      </div>
    `;
  });
}

function renderAdminCheckpoint(state) {
  const container = document.getElementById('admin-checkpoint-details');
  if (!container) return;

  container.innerHTML = `
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px;">
      <div style="background: var(--bg-canvas); padding: 12px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
        <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase;">Status</div>
        <div style="font-size: 15px; font-weight: 700; color: ${state.status === 'IDLE' ? 'var(--color-success)' : 'var(--color-warning)'}">${state.status}</div>
      </div>
      <div style="background: var(--bg-canvas); padding: 12px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
        <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase;">Cycle ID</div>
        <div style="font-size: 15px; font-weight: 700; color: var(--text-primary);">${state.cycle_id || 'None'}</div>
      </div>
      <div style="background: var(--bg-canvas); padding: 12px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
        <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase;">Retries</div>
        <div style="font-size: 15px; font-weight: 700; color: var(--text-primary);">${state.retry_count || 0}</div>
      </div>
      <div style="background: var(--bg-canvas); padding: 12px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle);">
        <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase;">Last Error</div>
        <div style="font-size: 13px; font-weight: 600; color: var(--color-danger); text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${state.last_error || 'None'}</div>
      </div>
    </div>
  `;
}

function renderRecentHistory(history) {
  const tbody = document.getElementById('admin-history-tbody');
  if (!tbody) return;

  if (history.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 24px;">No published drafts yet.</td></tr>';
    return;
  }

  tbody.innerHTML = '';
  history.forEach(item => {
    const snippet = (item.post_content || '').substring(0, 75) + '...';
    tbody.innerHTML += `
      <tr>
        <td><strong>#${item.cycle_id || '-'}</strong></td>
        <td>${new Date(item.completed_at).toLocaleString()}</td>
        <td><span style="padding: 3px 8px; border-radius: 9999px; background: rgba(16, 185, 129, 0.15); color: var(--color-success); font-weight: 700; font-size: 11px;">Drafted</span></td>
        <td style="font-family: var(--font-mono); font-size: 12px;">${item.buffer_update_id || '-'}</td>
        <td style="color: var(--text-secondary);">${snippet}</td>
      </tr>
    `;
  });
}

function refreshAdminData() {
  refreshPipelineStatus();
  loadBufferProfiles();
}

async function pingProvider(providerName) {
  const resultEl = document.getElementById(`ping-result-${providerName}`);
  if (resultEl) {
    resultEl.innerHTML = '<span class="spinner" style="display: inline-block;"></span>';
  }

  try {
    const res = await fetch('/api/test-provider', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: providerName })
    });
    const data = await res.json();
    if (data.success) {
      if (resultEl) {
        resultEl.innerHTML = `<span style="color: var(--color-success); font-weight: 700;">✓ ${data.latency_seconds.toFixed(2)}s (${data.model})</span>`;
      }
      showToast(`✓ ${providerName.toUpperCase()} online (${data.latency_seconds.toFixed(2)}s)`, 'success');
    } else {
      if (resultEl) {
        resultEl.innerHTML = `<span style="color: var(--color-danger); font-weight: 700;">❌ Error</span>`;
      }
      showToast(`Test failed: ${data.error}`, 'error');
    }
  } catch (err) {
    if (resultEl) {
      resultEl.innerHTML = `<span style="color: var(--color-danger);">❌ ${err.message}</span>`;
    }
  }
}

async function checkCloudinary() {
  const resultEl = document.getElementById('ping-result-cloudinary');
  if (resultEl) {
    resultEl.innerHTML = '<span class="spinner" style="display: inline-block;"></span>';
  }
  try {
    const res = await fetch('/api/cloudinary');
    const data = await res.json();
    if (resultEl) {
      if (data.configured) {
        resultEl.innerHTML = `<span style="color: var(--color-success); font-weight: 700;">✓ Active (${data.cloud_name || 'CDN'})</span>`;
      } else {
        resultEl.innerHTML = '<span style="color: var(--color-warning); font-weight: 600;">⚠️ Not Configured</span>';
      }
    }
  } catch (err) {
    if (resultEl) {
      resultEl.innerHTML = `<span style="color: var(--color-danger);">❌ ${err.message}</span>`;
    }
  }
}

// -----------------------------------------------------------------------------
// Toast Notifications
// -----------------------------------------------------------------------------
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${message}</span>`;

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 250);
  }, 4000);
}
