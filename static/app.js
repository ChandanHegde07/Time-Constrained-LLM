/**
 * Time Constrained LLM Testing System - Frontend JavaScript
 */

class ExperimentApp {
    constructor() {
        this.statusInterval = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.updateStatus();
    }

    bindEvents() {
        // Start button
        document.getElementById('startBtn').addEventListener('click', () => this.startExperiment());
        
        // Stop button
        document.getElementById('stopBtn').addEventListener('click', () => this.stopExperiment());
        
        // Reset button
        document.getElementById('resetBtn').addEventListener('click', () => this.resetForm());
        
        // Clear logs button
        document.getElementById('clearLogsBtn').addEventListener('click', () => this.clearLogs());
        
        // Export buttons
        document.getElementById('exportJsonBtn').addEventListener('click', () => this.exportJson());
        document.getElementById('exportCsvBtn').addEventListener('click', () => this.exportCsv());
        
        // Slider value update
        const pressureSlider = document.getElementById('time_pressure_ratio');
        const pressureValue = document.getElementById('pressure_value');
        pressureSlider.addEventListener('input', (e) => {
            pressureValue.textContent = e.target.value;
        });
    }

    getConfig() {
        // Get selected time limits
        const timeLimits = Array.from(document.querySelectorAll('input[name="time_limits"]:checked'))
            .map(cb => parseFloat(cb.value));
        
        // Get selected task categories
        const taskCategories = Array.from(document.querySelectorAll('input[name="task_categories"]:checked'))
            .map(cb => cb.value);
        
        return {
            experiment_name: document.getElementById('experiment_name').value || 'web_experiment',
            task_count: parseInt(document.getElementById('task_count').value) || 10,
            time_limits: timeLimits.length > 0 ? timeLimits : [1.0, 2.0, 3.0, 5.0, 10.0],
            task_categories: taskCategories.length > 0 ? taskCategories : ['reasoning', 'creative', 'qa', 'analytical', 'problem_solving'],
            difficulty_levels: [1, 2, 3, 4, 5],
            time_pressure_ratio: parseFloat(document.getElementById('time_pressure_ratio').value) || 0.6,
            quality_scoring_enabled: document.getElementById('quality_scoring_enabled').checked,
            statistical_analysis_enabled: document.getElementById('statistical_analysis_enabled').checked,
            output_formats: ['json'],
            log_level: document.getElementById('log_level').value,
            timeout: 600.0
        };
    }

    async startExperiment() {
        const config = this.getConfig();
        
        if (config.task_categories.length === 0) {
            this.addLog('error', 'Please select at least one task category');
            return;
        }
        
        if (config.time_limits.length === 0) {
            this.addLog('error', 'Please select at least one time limit');
            return;
        }

        this.setButtonsState(true);
        this.addLog('info', `Starting experiment: ${config.experiment_name}`);
        this.addLog('info', `Tasks: ${config.task_count}, Time Pressure: ${config.time_pressure_ratio}`);

        try {
            const response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            const data = await response.json();

            if (response.ok) {
                this.addLog('success', 'Experiment started successfully');
                this.startStatusPolling();
            } else {
                this.addLog('error', data.error || 'Failed to start experiment');
                this.setButtonsState(false);
            }
        } catch (error) {
            this.addLog('error', `Error: ${error.message}`);
            this.setButtonsState(false);
        }
    }

    async stopExperiment() {
        try {
            const response = await fetch('/api/stop', {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.addLog('warning', 'Experiment stopped by user');
                this.stopStatusPolling();
            } else {
                this.addLog('error', data.error || 'Failed to stop experiment');
            }
        } catch (error) {
            this.addLog('error', `Error: ${error.message}`);
        }
    }

    resetForm() {
        document.getElementById('experimentForm').reset();
        document.getElementById('pressure_value').textContent = '0.6';
        
        // Re-check default values
        document.querySelectorAll('input[name="time_limits"][value="3.0"], input[name="time_limits"][value="5.0"], input[name="time_limits"][value="10.0"]').forEach(cb => cb.checked = true);
        document.querySelectorAll('input[name="task_categories"]').forEach(cb => cb.checked = true);
        document.getElementById('quality_scoring_enabled').checked = true;
        document.getElementById('statistical_analysis_enabled').checked = true;
        
        this.addLog('info', 'Form reset to defaults');
    }

    startStatusPolling() {
        this.statusInterval = setInterval(() => this.updateStatus(), 1000);
    }

    stopStatusPolling() {
        if (this.statusInterval) {
            clearInterval(this.statusInterval);
            this.statusInterval = null;
        }
    }

    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const state = await response.json();

            // Update status indicator
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            statusDot.className = 'status-dot ' + state.status;
            statusText.textContent = this.getStatusText(state.status);

            // Update progress
            const progress = state.progress || 0;
            document.getElementById('progressFill').style.width = progress + '%';
            document.getElementById('progressText').textContent = Math.round(progress) + '%';

            // Update stats
            const stats = state.statistics || {};
            document.getElementById('totalTasks').textContent = state.total_tasks || 0;
            document.getElementById('completedTasks').textContent = (stats.completed_tasks || 0);
            document.getElementById('timedOutTasks').textContent = (stats.timed_out_tasks || 0);
            document.getElementById('errorTasks').textContent = (stats.error_tasks || 0);

            // Update duration
            document.getElementById('duration').textContent = (state.duration || 0).toFixed(1) + 's';

            // Handle completion
            if (state.status === 'completed' || state.status === 'error' || state.status === 'stopped') {
                this.stopStatusPolling();
                this.setButtonsState(false);
                
                if (state.status === 'completed') {
                    this.addLog('success', 'Experiment completed!');
                    this.loadResults();
                } else if (state.status === 'error') {
                    this.addLog('error', `Experiment failed: ${state.error}`);
                }
            }

            // Update logs if there are new ones
            if (state.logs && state.logs.length > 0) {
                const logsContainer = document.getElementById('logsContainer');
                const lastLog = state.logs[state.logs.length - 1];
                const existingLogs = logsContainer.querySelectorAll('.log-entry');
                const lastExistingLog = existingLogs[existingLogs.length - 1];
                
                if (!lastExistingLog || lastExistingLog.textContent !== lastLog.message) {
                    this.addLog('info', lastLog.message);
                }
            }
        } catch (error) {
            console.error('Error fetching status:', error);
        }
    }

    getStatusText(status) {
        const statusMap = {
            'idle': 'Ready',
            'running': 'Running...',
            'completed': 'Completed',
            'error': 'Error',
            'stopped': 'Stopped'
        };
        return statusMap[status] || status;
    }

    setButtonsState(isRunning) {
        document.getElementById('startBtn').disabled = isRunning;
        document.getElementById('stopBtn').disabled = !isRunning;
        
        // Disable form inputs while running
        const formInputs = document.querySelectorAll('#experimentForm input, #experimentForm select');
        formInputs.forEach(input => input.disabled = isRunning);
    }

    addLog(level, message) {
        const logsContainer = document.getElementById('logsContainer');
        const timestamp = new Date().toLocaleTimeString();
        
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${level}`;
        logEntry.innerHTML = `<span class="timestamp">[${timestamp}]</span> ${message}`;
        
        logsContainer.appendChild(logEntry);
        logsContainer.scrollTop = logsContainer.scrollHeight;
        
        // Limit log entries
        while (logsContainer.children.length > 100) {
            logsContainer.removeChild(logsContainer.firstChild);
        }
    }

    clearLogs() {
        const logsContainer = document.getElementById('logsContainer');
        logsContainer.innerHTML = '<div class="log-entry system">Logs cleared</div>';
    }

    async loadResults() {
        try {
            console.log('Loading results...');
            const response = await fetch('/api/results');
            const data = await response.json();
            console.log('Results data:', data);
            
            if (data.results && data.results.length > 0) {
                this.displayResults(data.results, data.statistics);
                document.getElementById('exportJsonBtn').disabled = false;
                document.getElementById('exportCsvBtn').disabled = false;
            } else {
                console.log('No results found or results empty');
            }
        } catch (error) {
            console.error('Error loading results:', error);
            this.addLog('error', `Error loading results: ${error.message}`);
        }
    }

    displayResults(results, statistics) {
        // Show summary
        const summaryEl = document.getElementById('resultsSummary');
        summaryEl.style.display = 'block';
        
        if (statistics) {
            document.getElementById('statTotalTasks').textContent = statistics.total_tasks || 0;
            
            const completionRate = statistics.total_tasks > 0 
                ? ((statistics.completed_tasks / statistics.total_tasks) * 100).toFixed(1) + '%'
                : '0%';
            document.getElementById('statCompletionRate').textContent = completionRate;
            
            document.getElementById('statAvgResponseTime').textContent = 
                statistics.avg_response_time ? statistics.avg_response_time.toFixed(2) + 's' : '-';
            
            document.getElementById('statAvgQuality').textContent = 
                statistics.avg_quality_score ? statistics.avg_quality_score.toFixed(1) : '-';
            
            document.getElementById('statAvgOutput').textContent = 
                statistics.avg_output_length ? Math.round(statistics.avg_output_length) + ' chars' : '-';
            
            const successRate = statistics.total_tasks > 0
                ? ((statistics.completed_tasks / statistics.total_tasks) * 100).toFixed(1) + '%'
                : '0%';
            document.getElementById('statSuccessRate').textContent = successRate;
        }

        // Populate table
        const tbody = document.getElementById('resultsBody');
        tbody.innerHTML = '';

        results.forEach(result => {
            const row = document.createElement('tr');
            
            const statusClass = this.getStatusClass(result.status);
            const qualityScore = result.metrics?.quality_score?.toFixed(1) || '-';
            
            row.innerHTML = `
                <td>${result.task_id}</td>
                <td>${result.task_type}</td>
                <td>${result.time_limit}s</td>
                <td>${result.time_elapsed?.toFixed(2) || '-'}s</td>
                <td><span class="status-badge ${statusClass}">${result.status}</span></td>
                <td>${qualityScore}</td>
            `;
            
            tbody.appendChild(row);
        });
    }

    getStatusClass(status) {
        const statusMap = {
            'completed': 'completed',
            'success': 'completed',
            'timeout': 'timeout',
            'timed_out': 'timeout',
            'error': 'error',
            'pending': 'pending'
        };
        return statusMap[status?.toLowerCase()] || 'pending';
    }

    async exportJson() {
        try {
            const response = await fetch('/api/results');
            const data = await response.json();
            
            const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
            this.downloadBlob(blob, 'experiment_results.json');
            this.addLog('info', 'Results exported as JSON');
        } catch (error) {
            this.addLog('error', `Error exporting JSON: ${error.message}`);
        }
    }

    async exportCsv() {
        try {
            const response = await fetch('/api/results');
            const data = await response.json();
            
            if (!data.results || data.results.length === 0) {
                this.addLog('error', 'No results to export');
                return;
            }

            // Build CSV
            const headers = ['ID', 'Type', 'Time Limit', 'Elapsed', 'Status', 'Quality Score', 'Output'];
            const rows = data.results.map(r => [
                r.task_id,
                r.task_type,
                r.time_limit,
                r.time_elapsed?.toFixed(2) || '',
                r.status,
                r.metrics?.quality_score?.toFixed(1) || '',
                (r.output || '').replace(/"/g, '""')
            ]);

            let csv = headers.join(',') + '\n';
            rows.forEach(row => {
                csv += row.map(cell => `"${cell}"`).join(',') + '\n';
            });

            const blob = new Blob([csv], { type: 'text/csv' });
            this.downloadBlob(blob, 'experiment_results.csv');
            this.addLog('info', 'Results exported as CSV');
        } catch (error) {
            this.addLog('error', `Error exporting CSV: ${error.message}`);
        }
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ExperimentApp();
});
