document.addEventListener('DOMContentLoaded', () => {
    // ═══ DOM REFS ═══
    const $ = id => document.getElementById(id);
    
    // Global Elements
    const codeInput = $('code-input');
    const codeRender = $('code-render');
    const codeOverlay = $('code-render-overlay');
    const analyzeBtn = $('analyze-btn');
    const statusDot = $('status-dot');
    const statusText = $('status-text');

    // Sidebar Stats
    const scoreNumber = $('score-number');
    const statTime = $('stat-time');
    const statLoc = $('stat-loc');
    const statEnergy = $('stat-energy');
    const statCo2 = $('stat-co2');

    // Result Tabs
    const tabBtns = document.querySelectorAll('.res-tab');
    const tabPanels = document.querySelectorAll('.tab-panel');

    // Action Center
    const actionDock = $('action-center');
    const btnLocalize = $('btn-localize');
    const btnHeal = $('btn-heal-source');
    const btnPipeline = $('btn-pipeline');

    // Sub-Views
    const locInstructions = $('loc-instructions');
    const healHeader = $('heal-header');
    const diffBefore = $('diff-before');
    const diffAfter = $('diff-after');
    const logContainer = $('log-container');

    let currentCode = '';
    let detectedCWEs = [];

    // ═══ CORE LOGIC ═══
    function switchTab(tabId) {
        tabBtns.forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
        tabPanels.forEach(p => p.classList.toggle('active', p.id === tabId));
    }

    function showState(prefix, name) {
        ['idle', 'loading', 'success', 'error'].forEach(s => {
            const el = $(`${prefix}-${s}`);
            if (el) el.classList.add('hidden');
        });
        const target = $(`${prefix}-${name}`);
        if (target) target.classList.remove('hidden');
    }

    function setEditorMode(mode) {
        if (mode === 'edit') {
            codeOverlay.classList.add('hidden');
            codeInput.classList.remove('hidden');
            $('btn-edit').classList.add('hidden');
        } else {
            codeOverlay.classList.remove('hidden');
            codeInput.classList.add('hidden');
            $('btn-edit').classList.remove('hidden');
        }
    }

    function escapeHtml(str) {
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // ═══ EVENTS ═══
    tabBtns.forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));
    $('btn-clear').addEventListener('click', () => {
        codeInput.value = '';
        statLoc.textContent = '0';
        setEditorMode('edit');
        showState('state', 'idle');
    });
    $('btn-edit').addEventListener('click', () => setEditorMode('edit'));

    analyzeBtn.addEventListener('click', async () => {
        const code = codeInput.value.trim();
        if (!code) return;

        currentCode = code;
        showState('state', 'loading');
        statusDot.className = 'status-dot active';
        statusText.textContent = 'Analyzing...';

        try {
            const demo = window.getDemoResults ? window.getDemoResults(code) : null;
            let data;
            
            if (demo) {
                await new Promise(r => setTimeout(r, 1200));
                data = demo.analysis;
            } else {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: code })
                });
                data = await response.json();
            }

            if (data.success) {
                renderResults(data, code);
                showState('state', 'success');
                statusDot.className = 'status-dot done';
                statusText.textContent = 'Complete';
            } else {
                throw new Error(data.error);
            }
        } catch (err) {
            $('error-message').textContent = err.message;
            showState('state', 'error');
            statusDot.className = 'status-dot error';
            statusText.textContent = 'Error';
        }
    });

    function renderResults(data, code) {
        statTime.textContent = `${data.elapsed.toFixed(2)}s`;
        const lines = code.split('\n').filter(l => l.trim()).length;
        statLoc.textContent = lines;

        const preds = data.predictions || [];
        const detected = preds.filter(p => p.detected && p.class !== 'non-vulnerable');
        detectedCWEs = detected.map(p => p.class);

        // Score Calculation
        let highestVuln = 0;
        preds.forEach(p => {
            if(p.class !== 'non-vulnerable') highestVuln = Math.max(highestVuln, p.probability);
        });
        const score = detected.length > 0 ? Math.min(48, Math.round(100 - (highestVuln*100))) : Math.round(100 - (highestVuln*100));
        scoreNumber.textContent = score;
        scoreNumber.style.color = score > 80 ? 'var(--success)' : (score > 50 ? 'var(--warning)' : 'var(--danger)');

        // Verdict Card
        const banner = $('verdict-banner');
        if (detected.length > 0) {
            banner.className = 'verdict-card vuln';
            banner.innerHTML = `<h4>Vulnerability Detected</h4><p>Isolated ${detected.length} structural risks. Deep audit recommended.</p>`;
            actionDock.classList.remove('hidden');
        } else {
            banner.className = 'verdict-card safe';
            banner.innerHTML = `<h4>Verification Successful</h4><p>Code logic matches known-safe structural patterns.</p>`;
            actionDock.classList.add('hidden');
        }

        // Probability Stacks
        const probList = $('prob-list');
        probList.innerHTML = '';
        preds.sort((a,b) => b.probability - a.probability).forEach(p => {
            const isDanger = p.class !== 'non-vulnerable' && p.detected;
            const item = document.createElement('div');
            item.className = `prob-item ${isDanger ? 'danger' : ''}`;
            item.innerHTML = `
                <div class="prob-header">
                    <span class="prob-name">${p.class}</span>
                    <span class="prob-pct">${(p.probability*100).toFixed(1)}%</span>
                </div>
                <div class="p-bar"><div class="p-fill" style="width: ${p.probability*100}%"></div></div>
            `;
            probList.appendChild(item);
        });

        // Code Highlight
        if (window.highlightVulnerability && detected.length > 0) {
            codeRender.innerHTML = `<code>${window.highlightVulnerability(code, detected[0].function)}</code>`;
            setEditorMode('view');
        } else {
            codeRender.innerHTML = `<code>${escapeHtml(code)}</code>`;
            setEditorMode('view');
        }
    }

    // ═══ LOCALIZATION ═══
    btnLocalize.addEventListener('click', async () => {
        switchTab('tab-localization');
        $('loc-idle').classList.add('hidden');
        $('loc-loading').classList.remove('hidden');
        $('loc-results').classList.add('hidden');

        try {
            const demo = window.getDemoResults ? window.getDemoResults(currentCode) : null;
            let data;
            if (demo && demo.localization) {
                await new Promise(r => setTimeout(r, 800));
                data = { success: true, localizations: demo.localization };
            } else {
                const response = await fetch('/api/localize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: currentCode, cwes: detectedCWEs })
                });
                data = await response.json();
            }

            if (data.success) {
                $('loc-loading').classList.add('hidden');
                $('loc-results').classList.remove('hidden');
                locInstructions.innerHTML = '';
                const cwe = Object.keys(data.localizations)[0];
                data.localizations[cwe].forEach(inst => {
                    const el = document.createElement('div');
                    el.className = 'inst-item';
                    el.innerHTML = `
                        <div class="inst-head"><span class="inst-op">${inst.opcode}</span><span>${(inst.score*100).toFixed(1)}% weight</span></div>
                        <div class="inst-txt">${escapeHtml(inst.text)}</div>
                    `;
                    locInstructions.appendChild(el);
                });
            }
        } catch (e) { console.error(e); }
    });

    // ═══ HEALING ═══
    btnHeal.addEventListener('click', async () => {
        switchTab('tab-healing');
        $('heal-idle').classList.add('hidden');
        $('heal-results').classList.add('hidden');

        try {
            const demo = window.getDemoResults ? window.getDemoResults(currentCode) : null;
            let data;
            if (demo && demo.healed_source) {
                await new Promise(r => setTimeout(r, 900));
                data = { success: true, healed: true, healed_code: demo.healed_source, message: "AI Repair Patch Generated." };
            } else {
                const response = await fetch('/api/heal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ code: currentCode, mode: 'source' })
                });
                data = await response.json();
            }

            if (data.success && data.healed) {
                $('heal-results').classList.remove('hidden');
                healHeader.textContent = data.message;
                diffBefore.textContent = currentCode;
                diffAfter.textContent = data.healed_code;
                
                if (confirm("Heal complete. Apply patched code to editor?")) {
                    codeInput.value = data.healed_code;
                    setEditorMode('edit');
                }
            }
        } catch (e) { console.error(e); }
    });

    // ═══ PIPELINE ═══
    btnPipeline.addEventListener('click', async () => {
        logContainer.innerHTML = '<div class="log-entry active">> Initializing GNN Iterative Pipeline...</div>';
        
        try {
            const response = await fetch('/api/pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ code: currentCode, carbon_tracking: true })
            });
            const data = await response.json();
            
            if (data.success) {
                logContainer.innerHTML = '';
                data.iterations.forEach(iter => {
                    const line = document.createElement('div');
                    line.className = 'log-entry done';
                    line.textContent = `> Iteration ${iter.iteration}: Validation Passed`;
                    logContainer.appendChild(line);
                });
                const final = document.createElement('div');
                final.className = 'log-entry active';
                final.textContent = `> Final Status: ${data.status} | Total Patches: ${data.total_patches}`;
                logContainer.appendChild(final);

                if (data.carbon_metrics) {
                    statEnergy.textContent = data.carbon_metrics.energy_kwh.toFixed(4);
                    statCo2.textContent = data.carbon_metrics.co2_kg.toFixed(5);
                }
            }
        } catch (e) { console.error(e); }
    });
});
