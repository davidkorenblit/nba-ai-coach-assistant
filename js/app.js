// אובייקטים לניהול הגרפים
        let shapChart = null;
        let lookaheadChart = null;
        let roiChart = null;

        // מצב הזיכרון החי של האפליקציה (Session State)
        let dataset = null;
        let currentGame = "game_1";
        let currentIndex = 0;
        let isPlaying = false;
        let simTimeout = null;
        window.currentCrisisMuted = false;

        // === SIMULATION RUNTIME LOGGER ===
        window.SIM_LOG = [];
        window.breakoutTriggeredForCurrentAlert = false;

        const TEAMS = {
            game_1: { home: "IND", away: "TOR", homeFull: "Indiana Pacers", awayFull: "Toronto Raptors" },
            game_2: { home: "BOS", away: "MIA", homeFull: "Boston Celtics", awayFull: "Miami Heat" }
        };

        function updateTeamLabels() {
            const teamConfig = TEAMS[currentGame];
            document.getElementById('team-home-label').innerText = teamConfig.home;
            document.getElementById('team-home-name').innerText = teamConfig.homeFull;
            document.getElementById('team-away-label').innerText = teamConfig.away;
            document.getElementById('team-away-name').innerText = teamConfig.awayFull;
        }

        function exportSimLog() {
            if (!window.SIM_LOG || window.SIM_LOG.length === 0) {
                alert("אין נתוני סימולציה לייצא כרגע. אנא הרץ את הסימולציה תחילה.");
                return;
            }

            // Map structured runtime logs into clean, ticker-matching plain text lines
            const logLines = window.SIM_LOG.map(row => {
                const t = TEAMS[row.game || currentGame];
                return `[Q${row.period} - P:${row.possession_index}] ${t.home} ${row.home_score} : ${t.away} ${row.away_score} | ${row.play_description}`;
            });

            const textContent = logLines.join("\n");

            // Safe, user-triggered blob download policy
            const blob = new Blob([textContent], { type: "text/plain;charset=utf-8" });
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = `simulation_log_${currentGame}_${Date.now()}.txt`;

            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }

        // טעינת נתונים אוטומטית ממערך JS
        window.addEventListener('DOMContentLoaded', () => {
            if (window.DEMO_DATA) {
                dataset = window.DEMO_DATA;
                resetSimulation();
            } else {
                console.error("Dataset not found. Please ensure demo_data.js is loaded.");
                document.getElementById('file-picker-bar').classList.remove('hidden');
                document.getElementById('file-picker-bar').classList.add('flex');
            }
            initRoiChart();
        });

        // Handle manual file upload if JS dataset is not present
        function handleJsonUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function (e) {
                try {
                    dataset = JSON.parse(e.target.result);
                    document.getElementById('file-picker-bar').innerHTML = "<span class='text-emerald-400 font-bold'>✓ Data loaded successfully! The simulator is ready for use.</span>";
                    resetSimulation();
                } catch (err) {
                    alert("Error parsing JSON file. Make sure it is the correct file.");
                }
            };
            reader.readAsText(file);
        }

        // Switch views (Pages Layout Manager)
        function switchView(viewName) {
            document.getElementById('view-simcast').classList.add('hidden');
            document.getElementById('view-roi').classList.add('hidden');
            document.getElementById('tab-simcast').className = "px-4 py-1 rounded text-slate-400 hover:text-white font-bold transition";
            document.getElementById('tab-roi').className = "px-4 py-1 rounded text-slate-400 hover:text-white font-bold transition";

            if (viewName === 'simcast') {
                document.getElementById('view-simcast').classList.remove('hidden');
                document.getElementById('tab-simcast').className = "px-4 py-1 rounded bg-amber-500 text-slate-950 font-bold transition";
            } else if (viewName === 'roi') {
                document.getElementById('view-roi').classList.remove('hidden');
                document.getElementById('tab-roi').className = "px-4 py-1 rounded bg-amber-500 text-slate-950 font-bold transition";
            }
        }

        // Change game context (Game 1 or Game 2)
        function changeGameContext() {
            currentGame = document.getElementById('game-selector').value;
            resetSimulation();
        }

        // Reset simulation engine
        function resetSimulation() {
            pauseSimulation();
            currentIndex = 0;
            window.SIM_LOG = [];
            window.breakoutTriggeredForCurrentAlert = false;
            window.currentCrisisMuted = false;
            document.getElementById('live-ticker-feed').innerHTML = "<div class='text-amber-500 font-bold'>> Arena reset complete. Context changed to: " + currentGame + "</div>";
            updateTeamLabels();
            updateHudDisplay();
        }

        // Pause simulation
        function pauseSimulation() {
            isPlaying = false;
            if (simTimeout) clearTimeout(simTimeout);
            // מנקה שאריות מהגרסאות הישנות למקרה שקיים
            if (window.simInterval) clearInterval(window.simInterval);
        }

        // Play simulation - מתוקן עם setTimeout רקורסיבי לתמיכה בסליידר חי
        function playSimulation() {
            if (!dataset) {
                alert("אנא ודא שקובץ הנתונים נטען בהצלחה.");
                return;
            }
            if (isPlaying) return;
            isPlaying = true;

            const speedSlider = document.getElementById('speed-slider');

            function tick() {
                if (!isPlaying) return;

                const gameData = dataset[currentGame];
                if (currentIndex >= gameData.length - 1) {
                    pauseSimulation();
                    if (currentGame === "game_2") {
                        switchView('roi');
                    }
                    return;
                }

                currentIndex++;
                updateHudDisplay();
                checkTacticalTriggers(gameData[currentIndex]);

                // רק אם המשחק עדיין רץ (לא נעצר על ידי טיים אאוט או אלרט) קובעים את הטיק הבא
                if (isPlaying) {
                    const speed = speedSlider.max - speedSlider.value + 50;
                    simTimeout = setTimeout(tick, speed);
                }
            }

            // קריאה ראשונה ללולאה
            const initialSpeed = speedSlider.max - speedSlider.value + 50;
            simTimeout = setTimeout(tick, initialSpeed);
        }

        // עדכון שוטף של מדדי המשחק על המסך (HUD Update)
        function updateHudDisplay() {
            if (!dataset) return;
            const row = dataset[currentGame][currentIndex];

            document.getElementById('score-home').innerText = row.home_score;
            document.getElementById('score-away').innerText = row.away_score;
            document.getElementById('hud-period').innerText = row.period;
            document.getElementById('hud-margin').innerText = row.score_margin;
            document.getElementById('hud-possession').innerText = currentIndex;
            document.getElementById('home-fatigue').innerText = Math.round(row.home_cum_fatigue) + "s";
            document.getElementById('away-fatigue').innerText = Math.round(row.away_cum_fatigue) + "s";

            if (currentIndex % 3 === 0 || row.timeout_team !== "NONE" || row.target_stop_run_90s === 1) {
                const ticker = document.getElementById('live-ticker-feed');
                const pDesc = row.play_description || "Normal possession trade";
                ticker.insertAdjacentHTML('afterbegin', `<div>[Q${row.period} - P:${currentIndex}] ${TEAMS[currentGame].home} ${row.home_score} : ${TEAMS[currentGame].away} ${row.away_score} | ${pDesc}</div>`);
            }

            // עדכון לוג הריצה הכללי עבור כל פוזשן ופוזשן
            const logEntry = {
                timestamp_ms: Date.now(),
                simulation_clock_s: currentIndex * 1.5,
                game: currentGame,
                possession_index: currentIndex,
                period: row.period,
                home_score: row.home_score,
                away_score: row.away_score,
                score_margin: row.score_margin,
                cate_score: row.cate_score,
                propensity_score: row.propensity_score,
                alert_triggered: row.target_stop_run_90s === 1,
                timeout_called: row.timeout_team !== "NONE",
                timeout_team: row.timeout_team,
                play_description: row.play_description || "Normal possession trade",
                ui_event: null
            };
            window.SIM_LOG.push(logEntry);

            const sysLight = document.getElementById('light-system');
            const coachLight = document.getElementById('light-coach');

            if (row.cate_score > 0.6) {
                sysLight.className = "w-5 h-5 md:w-7 md:h-7 rounded-full led-indicator led-red-active";
            } else {
                sysLight.className = "w-5 h-5 md:w-7 md:h-7 rounded-full led-indicator";
            }

            if (row.propensity_score > 0.7) {
                coachLight.className = "w-5 h-5 md:w-7 md:h-7 rounded-full led-indicator led-green-active";
            } else {
                coachLight.className = "w-5 h-5 md:w-7 md:h-7 rounded-full led-indicator";
            }
        }

        // מאזין הטריגרים והאירועים - כולל מנגנון ה-Crisis Muting
        function checkTacticalTriggers(row) {
            // 1. Timeout requested banner (always show if timeout_team is not NONE)
            if (row.timeout_team !== "NONE") {
                triggerTimeoutBanner(row.timeout_team, row);
            }

            // 2. Momentum Crisis alert (System Alerts) or Q2 Presentation Specific Timeout
            if (row.target_stop_run_90s === 1 || (row.period === 2 && row.timeout_team !== "NONE")) {
                window.lastAlertRow = row;

                if (!window.currentCrisisMuted) {
                    pauseSimulation();
                    openTacticalBreakout(row);
                }
            } else {
                // Reset muting state when we exit the alert zone (crisis is over)
                window.currentCrisisMuted = false;

                // If it's a tactical timeout (timeout called, but not during a momentum crisis)
                if (row.timeout_team !== "NONE") {
                    triggerTacticalToast(row);
                }
            }
        }

        function triggerTacticalToast(row) {
            if (window.SIM_LOG.length > 0) {
                window.SIM_LOG[window.SIM_LOG.length - 1].ui_event = "TACTICAL_TOAST_SHOWN";
            }
            const toast = document.getElementById('toast-tactical');
            const textEl = document.getElementById('toast-tactical-text');
            const toastHeader = toast ? toast.querySelector('h4') : null;
            if (toast) {
                if (currentGame === 'game_2' && row && row.timeout_team === 'BOSTON') {
                    if (toastHeader) toastHeader.innerText = "Timeout Decision";
                    if (textEl) textEl.innerText = "Timeout Decision Logged";
                } else {
                    if (toastHeader) toastHeader.innerText = "Tactical Decision";
                    if (textEl) textEl.innerText = "Tactical Decision Logged";
                }
                toast.classList.remove('hidden');
            }
        }

        function closeTacticalToast() {
            document.getElementById('toast-tactical').classList.add('hidden');
        }

        function triggerTimeoutBanner(teamName, row) {
            pauseSimulation();
            const overlay = document.getElementById('timeout-overlay');
            let displayName = teamName;
            if (teamName === "INDIANA") displayName = "INDIANA PACERS";
            else if (teamName === "TORONTO") displayName = "TORONTO RAPTORS";
            else if (teamName === "BOSTON") displayName = "BOSTON CELTICS";
            else if (teamName === "MIAMI") displayName = "MIAMI HEAT";

            const bannerEl = document.getElementById('timeout-team-banner');
            if (bannerEl) bannerEl.innerText = displayName;

            const descEl = document.getElementById('timeout-desc-banner');
            if (descEl) {
                let descText = row && row.play_description ? row.play_description : "DSS AUTOMATIC CAPTURE PROCESS ACTIVE";
                const isGame1EarlyTimeout = (currentGame === 'game_1' && row && row.period <= 2);
                const isGame2BostonTimeout = (currentGame === 'game_2' && row && row.timeout_team === 'BOSTON');
                
                if (isGame1EarlyTimeout || isGame2BostonTimeout) {
                    descText = descText.replace(/tactical/gi, "").trim();
                    descText = descText.replace(/^[:\s\-]+/, "").trim();
                }
                descEl.innerText = descText.toUpperCase();
            }

            overlay.classList.remove('hidden');

            setTimeout(() => {
                overlay.classList.add('hidden');
                // Only resume if we are not currently paused by a breakout modal
                const breakoutOpen = !document.getElementById('modal-breakout').classList.contains('hidden');
                if (!breakoutOpen) {
                    playSimulation();
                }
            }, 2500);
        }

        function openTacticalBreakout(row, forcedMode) {
            if (window.SIM_LOG.length > 0) {
                window.SIM_LOG[window.SIM_LOG.length - 1].ui_event = `BREAKOUT_OPENED:${forcedMode || row.period}`;
            }

            document.getElementById('modal-breakout').classList.remove('hidden');
            const textArea = document.getElementById('breakout-text-area');
            const mainGrid = document.getElementById('breakout-main-grid');
            const lookaheadContainer = document.getElementById('breakout-lookahead-container');
            const shapContainer = document.getElementById('breakout-shap-container');

            // Reset layout and visibility
            if (mainGrid) mainGrid.className = "grid grid-cols-1 md:grid-cols-2 gap-4";
            if (lookaheadContainer) lookaheadContainer.classList.remove('hidden');
            if (shapContainer) shapContainer.classList.remove('hidden');

            if (row.period === 1) {
                // Q1: Custom HTML view without charts
                if (lookaheadContainer) lookaheadContainer.classList.add('hidden');
                if (shapContainer) shapContainer.classList.add('hidden');
                if (mainGrid) mainGrid.className = "grid grid-cols-1 gap-4";

                // Generate future timeline (static hardcoded data for pitch presentation)
                let futureHtml = '<div class="flex flex-wrap justify-center gap-3 mt-4">';
                let fakeData = [
                    { label: "Possession +1", home: 27, away: 18 },
                    { label: "Possession +2", home: 27, away: 20 },
                    { label: "Possession +3", home: 27, away: 21 },
                    { label: "Possession +4", home: 29, away: 21 },
                    { label: "Possession +5", home: 29, away: 24 }
                ];
                
                fakeData.forEach((item) => {
                    futureHtml += `<div class="bg-slate-900/80 border border-emerald-500/40 rounded-xl p-3 md:p-4 text-center shadow-lg min-w-[90px]">
                        <span class="text-[10px] text-slate-400 block uppercase tracking-wider mb-1">${item.label}</span>
                        <span class="text-xl md:text-2xl font-digital text-white drop-shadow-md">${item.home} : ${item.away}</span>
                    </div>`;
                });
                futureHtml += '</div>';

                textArea.innerHTML = `
                    <div class="text-center py-6 space-y-6">
                        <h3 class="text-3xl font-black text-rose-500 tracking-wider uppercase animate-pulse">CRITICAL RUN DETECTED</h3>
                        <div class="text-5xl font-digital text-white py-4 drop-shadow-lg">${row.home_score} : ${row.away_score}</div>
                        
                        <div class="flex flex-wrap justify-center gap-3 mb-4">
                            <div class="inline-flex items-center gap-2 bg-slate-950/60 border border-slate-800 px-3 py-1 rounded-full text-xs font-semibold text-slate-300">
                                <span class="animate-pulse">🥅</span>
                                <span>Defense Decline: ${(row.shap_defensive_collapse || 0.0).toFixed(4)}</span>
                            </div>
                            <div class="inline-flex items-center gap-2 bg-slate-950/60 border border-slate-800 px-3 py-1 rounded-full text-xs font-semibold text-slate-300">
                                <span class="animate-pulse">🔋</span>
                                <span>Fatigue Stress: ${row.away_cum_fatigue > 0 ? (row.home_cum_fatigue / row.away_cum_fatigue).toFixed(2) : '1.00'}</span>
                            </div>
                        </div>

                        <p class="text-xl text-amber-400 font-bold uppercase tracking-wider">Primary Metric: Score Margin. 15-0 Run Detected.</p>
                        <p class="text-sm text-slate-400 max-w-lg mx-auto leading-relaxed">
                            At this early stage of the game, raw score is the dominant factor for the system's alert. The system prioritizes stopping the run before compounding stress factors begin to accumulate.
                        </p>
                        
                        <div class="pt-8 border-t border-slate-800/80 mt-8">
                            <button onclick="document.getElementById('q1-forecast').classList.remove('hidden'); this.classList.add('hidden');" class="bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-6 py-2 rounded-lg transition shadow-lg">
                                👁️ Reveal Simulation Forecast
                            </button>
                            <div id="q1-forecast" class="hidden transition-all duration-500">
                                <h4 class="text-emerald-400 font-bold tracking-widest uppercase mb-4 mt-2">Counterfactual Result: Bleeding Stopped</h4>
                                ${futureHtml}
                            </div>
                        </div>
                    </div>
                `;
            } else if (row.period === 2) {
                // Q2: Custom HTML view without charts
                if (lookaheadContainer) lookaheadContainer.classList.add('hidden');
                if (shapContainer) shapContainer.classList.add('hidden');
                if (mainGrid) mainGrid.className = "grid grid-cols-1 gap-4";
                
                // Generate future timeline (static hardcoded data for pitch presentation)
                let futureHtml = '<div class="flex flex-wrap justify-center gap-3 mt-4">';
                let fakeData = [
                    { label: "Possession +1", home: 46, away: 53 },
                    { label: "Possession +2", home: 48, away: 53 },
                    { label: "Possession +3", home: 48, away: 56 },
                    { label: "Possession +4", home: 51, away: 56 },
                    { label: "Possession +5", home: 51, away: 57 }
                ];
                
                fakeData.forEach((item) => {
                    futureHtml += `<div class="bg-slate-900/80 border border-emerald-500/40 rounded-xl p-3 md:p-4 text-center shadow-lg min-w-[90px]">
                        <span class="text-[10px] text-slate-400 block uppercase tracking-wider mb-1">${item.label}</span>
                        <span class="text-xl md:text-2xl font-digital text-white drop-shadow-md">${item.home} : ${item.away}</span>
                    </div>`;
                });
                futureHtml += '</div>';

                textArea.innerHTML = `
                    <div class="text-center py-6 space-y-6">
                        <div class="flex flex-wrap justify-center gap-3 mb-2">
                            <div class="inline-flex items-center gap-2 bg-slate-950/60 border border-slate-800 px-3 py-1 rounded-full text-xs font-semibold text-slate-300">
                                <span class="animate-pulse">🥅</span>
                                <span>Defense Decline: ${(row.shap_defensive_collapse || 0.0).toFixed(4)}</span>
                            </div>
                            <div class="inline-flex items-center gap-2 bg-slate-950/60 border border-slate-800 px-3 py-1 rounded-full text-xs font-semibold text-slate-300">
                                <span class="animate-pulse">🔋</span>
                                <span>Fatigue Stress: ${row.away_cum_fatigue > 0 ? (row.home_cum_fatigue / row.away_cum_fatigue).toFixed(2) : '1.00'}</span>
                            </div>
                        </div>
                        <h3 class="text-3xl font-black text-amber-500 tracking-wider uppercase">DYNAMIC MOMENTUM SHIFT</h3>
                        <div class="text-5xl font-digital text-white py-4 drop-shadow-lg">${row.home_score} : ${row.away_score}</div>
                        <p class="text-xl text-amber-400 font-bold uppercase tracking-wider">A shorter 6-0 run triggered the alert.</p>
                        <p class="text-sm text-slate-400 max-w-lg mx-auto leading-relaxed">
                            Momentum is flexible. As the game progresses, secondary factors such as fatigue start to accumulate. The system dynamically lowers its run detection threshold because the lineup's capacity to absorb pressure has decreased significantly compared to Q1.
                        </p>
                        
                        <div class="pt-8 border-t border-slate-800/80 mt-8">
                            <button onclick="document.getElementById('q2-forecast').classList.remove('hidden'); this.classList.add('hidden');" class="bg-emerald-600 hover:bg-emerald-500 text-white font-bold px-6 py-2 rounded-lg transition shadow-lg">
                                👁️ Reveal Tactical Impact
                            </button>
                            <div id="q2-forecast" class="hidden transition-all duration-500">
                                <h4 class="text-emerald-400 font-bold tracking-widest uppercase mb-4 mt-2">Timeout Impact: Crisis Averted</h4>
                                ${futureHtml}
                            </div>
                        </div>
                    </div>
                `;
            } else {
                // Q3: Deep Dive layout with charts
                if (currentGame === 'game_1') {
                    textArea.innerHTML = `
                        <div class="space-y-4 text-xs">
                            <p><b>Extreme state and system neglect:</b> A terrible long run was built by the opponent. The pressure in the arena is at its peak.</p>
                            <p><b>System Indicator:</b> The CATE metric has crossed the extreme Top 5% threshold. The red light is flashing wildly.</p>
                            <p><b>Human Failure:</b> The coach froze under mental pressure and did not take a timeout in reality. Notably, the propensity model had already predicted multiple times prior to this that a standard coach would call a timeout (the Green Light was ON, signaling early opportunities to intervene, which were all ignored).</p>
                            <p class='text-rose-400 font-bold mt-4'>>> The system precisely identified the tactical breakpoint and human blindness.</p>
                        </div>
                    `;
                } else {
                    textArea.innerHTML = `
                        <div class="space-y-4 text-xs">
                            <p><b>Immediate Reaction:</b> The coach reacted proactively, calling a timeout at possession 37, just one possession after the system's critical Red Alarm at possession 36.</p>
                            <p><b>Threat Analysis (SHAP):</b> The features on the right represent the high-risk factors (fatigue, defensive collapse, stale lineup) that were building up and would have caused a catastrophic collapse if ignored.</p>
                            <p><b>Avoided Catastrophe:</b> The lookahead chart shows that without this intervention (Red Line), Boston was projected to bleed out to a -15 deficit. By calling the timeout, this disaster was avoided, starting the comeback.</p>
                            <p class='text-emerald-400 font-bold mt-4'>>> The system proved the high ROI of stopping the run early, turning a near-catastrophe into a 2-point victory.</p>
                        </div>
                    `;
                }
                let mode = forcedMode;
                if (!mode) {
                    mode = (currentGame === 'game_1' && row.period === 3) ? "false_alarm" : "long_term";
                }
                initBreakoutCharts(mode, row);
            }
        }

        function closeBreakout() {
            document.getElementById('modal-breakout').classList.add('hidden');
            window.currentCrisisMuted = true;
            
            if (currentGame === 'game_1' && window.lastAlertRow && window.lastAlertRow.period <= 2) {
                const team = window.lastAlertRow.timeout_team !== "NONE" ? window.lastAlertRow.timeout_team : "INDIANA";
                triggerTimeoutBanner(team, window.lastAlertRow);
            } else {
                playSimulation();
            }
        }

        function openScoreFocus(row, eventNum) {
            const modal = document.getElementById('modal-score-focus');
            document.getElementById('score-focus-home-team').innerText = TEAMS[currentGame].home;
            document.getElementById('score-focus-home').innerText = row.home_score;
            document.getElementById('score-focus-away-team').innerText = TEAMS[currentGame].away;
            document.getElementById('score-focus-away').innerText = row.away_score;

            const title = document.getElementById('score-focus-title');
            const subtitle = document.getElementById('score-focus-subtitle');
            const forecastText = document.getElementById('score-focus-forecast-text');

            if (eventNum === 1) {
                title.innerText = "CRITICAL RUN DETECTED";
                subtitle.innerText = "15-0 Unanswered Points";
                forecastText.innerText = "Without intervention, momentum projects the margin to increase significantly. The scoreboard is the primary failure indicator at this early stage.";
            } else if (eventNum === 2) {
                title.innerText = "MOMENTUM SHIFT DETECTED";
                subtitle.innerText = "Shorter Run, High Risk";
                forecastText.innerText = "Momentum is context-dependent. As the game progresses, secondary factors amplify the risk. A smaller point run now carries the same mathematical danger as a 15-0 run in Q1.";
            }
            modal.classList.remove('hidden');
        }

        function closeScoreFocus() {
            document.getElementById('modal-score-focus').classList.add('hidden');
            window.currentCrisisMuted = true;
            playSimulation();
        }

        function initBreakoutCharts(mode, row) {
            const ctxShap = document.getElementById('chart-shap').getContext('2d');
            if (shapChart) shapChart.destroy();

            let shapLabels = ["Stale Lineup (Fatigue)", "Defensive Collapse", "Explosiveness Index", "Cumulative Fatigue"];
            let shapData = [
                row.shap_stale_lineup || 0.1,
                row.shap_defensive_collapse || 0.1,
                row.shap_explosiveness || 0.1,
                row.shap_fatigue || 0.1
            ];

            let shapColor = '#38bdf8';

            shapChart = new Chart(ctxShap, {
                type: 'bar',
                data: {
                    labels: shapLabels,
                    datasets: [{
                        data: shapData,
                        backgroundColor: shapColor,
                        borderWidth: 0
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, suggestedMax: 1.0 },
                        y: { ticks: { color: '#f1f5f9' } }
                    }
                }
            });

            const ctxLook = document.getElementById('chart-lookahead').getContext('2d');
            if (lookaheadChart) lookaheadChart.destroy();

            let labelTimeline = ["0s", "30s", "60s", "90s", "120s", "150s", "180s"];
            let realityData = mode === "short_term" ? [-2, -6, -10, -14, -14, -12, -10] : [0, -3, -6, -9, -11, -13, -15];
            let counterData = mode === "short_term" ? [-2, -2, 0, +2, +1, 0, -1] : [0, 0, -1, -2, -2, -3, -3];

            if (mode === "false_alarm") {
                realityData = [0, -2, -3, -1, 0, +1, +1];
                counterData = [0, 0, 0, 0, 0, 0, 0];
            }

            let realLabel = 'In Reality (Ignored)';
            let counterLabel = 'Model Simulation (Counterfactual)';
            if (currentGame === 'game_2') {
                realLabel = 'Model Simulation (If Ignored)';
                counterLabel = 'In Reality (Timeout Taken)';
            }

            lookaheadChart = new Chart(ctxLook, {
                type: 'line',
                data: {
                    labels: labelTimeline,
                    datasets: [
                        { label: realLabel, data: realityData, borderColor: '#ef4444', tension: 0.2, borderWidth: 2.5 },
                        { label: counterLabel, data: counterData, borderColor: '#22c55e', tension: 0.2, borderWidth: 2.5 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#f1f5f9' } } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        }

        function initRoiChart() {
            const ctxRoi = document.getElementById('chart-roi').getContext('2d');
            if (roiChart) roiChart.destroy();

            roiChart = new Chart(ctxRoi, {
                type: 'bar',
                data: {
                    labels: ["Stop Run 90s", "Improve Margin 90s", "Reverse Trend 180s", "Improve Margin 180s"],
                    datasets: [{
                        data: [3.2, 1.8, -0.9, -1.5],
                        backgroundColor: ['#22c55e', '#86efac', '#ef4444', '#b91c1c']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { grid: { display: false }, ticks: { color: '#f1f5f9' } },
                        y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                    }
                }
            });
        }