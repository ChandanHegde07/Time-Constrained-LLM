import os
import sys
import json
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from run_pipeline import ExperimentConfig, ExperimentRunner, ExperimentStatus

app = Flask(__name__)
CORS(app)

# Store experiment state
experiment_state = {
    "status": "idle",
    "progress": 0,
    "current_task": 0,
    "total_tasks": 0,
    "results": [],
    "logs": [],
    "start_time": None,
    "end_time": None,
    "error": None
}

# Lock for thread-safe access
state_lock = threading.Lock()

def log_message(message):
    """Add log message to experiment state"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    with state_lock:
        experiment_state["logs"].append({
            "timestamp": timestamp,
            "message": message
        })
        # Keep only last 100 logs
        if len(experiment_state["logs"]) > 100:
            experiment_state["logs"] = experiment_state["logs"][-100:]

def run_experiment_thread(config_dict):
    """Run experiment in background thread"""
    global experiment_state
    
    try:
        with state_lock:
            experiment_state["status"] = "running"
            experiment_state["start_time"] = time.time()
            experiment_state["logs"] = []
            experiment_state["error"] = None
        
        log_message(f"Starting experiment: {config_dict.get('experiment_name', 'default')}")
        
        # Create experiment config
        config = ExperimentConfig(
            experiment_name=config_dict.get("experiment_name", "web_experiment"),
            task_count=config_dict.get("task_count", 10),
            time_limits=config_dict.get("time_limits", [1.0, 2.0, 3.0, 5.0, 10.0]),
            task_categories=config_dict.get("task_categories", ["reasoning", "creative", "qa", "analytical", "problem_solving"]),
            difficulty_levels=config_dict.get("difficulty_levels", [1, 2, 3, 4, 5]),
            time_pressure_ratio=config_dict.get("time_pressure_ratio", 0.6),
            quality_scoring_enabled=config_dict.get("quality_scoring_enabled", True),
            statistical_analysis_enabled=config_dict.get("statistical_analysis_enabled", True),
            output_formats=config_dict.get("output_formats", ["json"]),
            log_level=config_dict.get("log_level", "INFO"),
            timeout=config_dict.get("timeout", 600.0)
        )
        
        # Force sequential execution to avoid API rate limiting
        from config import get_config
        cfg = get_config()
        cfg.resources.max_concurrent_requests = 1
        
        with state_lock:
            experiment_state["total_tasks"] = config.task_count
        
        # Run experiment
        runner = ExperimentRunner(config)
        result = runner.run()
        
        # Store results
        with state_lock:
            experiment_state["status"] = "completed"
            experiment_state["end_time"] = time.time()
            experiment_state["progress"] = 100
            experiment_state["current_task"] = config.task_count
            
            # Convert results to serializable format
            results_list = []
            for eval_result in result.results:
                results_list.append({
                    "task_id": eval_result.task_id,
                    "task_type": eval_result.task_type,
                    "time_limit": eval_result.time_limit,
                    "time_elapsed": eval_result.response.time_elapsed if eval_result.response else None,
                    "status": eval_result.response.status.value if eval_result.response and hasattr(eval_result.response.status, 'value') else str(eval_result.response.status if eval_result.response else 'unknown'),
                    "metrics": eval_result.metrics,
                    "output": eval_result.response.content[:500] if eval_result.response and eval_result.response.content else ""
                })
            
            experiment_state["results"] = results_list
            
            # Calculate average quality score from the quality_scores dict
            avg_quality = 0
            if result.statistics and result.statistics.quality_scores:
                quality_vals = list(result.statistics.quality_scores.values())
                avg_quality = sum(quality_vals) / len(quality_vals) if quality_vals else 0
            
            experiment_state["statistics"] = {
                "total_tasks": result.statistics.total_tasks if result.statistics else 0,
                "completed_tasks": result.statistics.completed_tasks if result.statistics else 0,
                "timed_out_tasks": result.statistics.timed_out_tasks if result.statistics else 0,
                "error_tasks": result.statistics.error_tasks if result.statistics else 0,
                "avg_completion_rate": result.statistics.avg_completion_rate if result.statistics else 0,
                "avg_response_time": result.statistics.avg_response_time if result.statistics else 0,
                "avg_quality_score": avg_quality,
                "avg_output_length": result.statistics.avg_output_length if result.statistics else 0
            }
        
        duration = experiment_state["end_time"] - experiment_state["start_time"]
        log_message(f"Experiment completed in {duration:.2f} seconds")
        log_message(f"Completed: {result.statistics.completed_tasks if result.statistics else 0}/{config.task_count} tasks")
        
    except Exception as e:
        with state_lock:
            experiment_state["status"] = "error"
            experiment_state["end_time"] = time.time()
            experiment_state["error"] = str(e)
        log_message(f"Error: {str(e)}")

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_default_config():
    """Get default configuration options"""
    return jsonify({
        "experiment_name": "web_experiment",
        "task_count": 10,
        "time_limits": [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 30.0],
        "task_categories": ["reasoning", "creative", "qa", "analytical", "problem_solving"],
        "difficulty_levels": [1, 2, 3, 4, 5],
        "time_pressure_ratio": 0.6,
        "quality_scoring_enabled": True,
        "output_formats": ["json", "csv"],
        "log_level": "INFO"
    })

@app.route('/api/start', methods=['POST'])
def start_experiment():
    """Start a new experiment"""
    global experiment_state
    
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        config = request.get_json()
        if config is None:
            return jsonify({"error": "Invalid JSON data"}), 400
    except Exception as e:
        return jsonify({"error": f"JSON parsing error: {str(e)}"}), 400
    
    with state_lock:
        if experiment_state["status"] == "running":
            return jsonify({"error": "Experiment already running"}), 400
        
        # Reset state
        experiment_state = {
            "status": "idle",
            "progress": 0,
            "current_task": 0,
            "total_tasks": 0,
            "results": [],
            "logs": [],
            "start_time": None,
            "end_time": None,
            "error": None
        }
    
    # Start experiment in background thread
    thread = threading.Thread(target=run_experiment_thread, args=(config,))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "started", "message": "Experiment started successfully"})

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current experiment status"""
    with state_lock:
        state = experiment_state.copy()
    
    # Calculate progress
    if state["total_tasks"] > 0:
        state["progress"] = (state["current_task"] / state["total_tasks"]) * 100
    
    if state["start_time"] and state["end_time"]:
        state["duration"] = state["end_time"] - state["start_time"]
    elif state["start_time"]:
        state["duration"] = time.time() - state["start_time"]
    else:
        state["duration"] = 0
    
    return jsonify(state)

@app.route('/api/stop', methods=['POST'])
def stop_experiment():
    """Stop the running experiment"""
    with state_lock:
        if experiment_state["status"] == "running":
            experiment_state["status"] = "stopped"
            experiment_state["end_time"] = time.time()
            log_message("Experiment stopped by user")
            return jsonify({"status": "stopped", "message": "Experiment stopped"})
        else:
            return jsonify({"error": "No experiment running"}), 400

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get experiment results"""
    with state_lock:
        return jsonify({
            "results": experiment_state["results"],
            "statistics": experiment_state.get("statistics", {}),
            "status": experiment_state["status"]
        })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get experiment logs"""
    with state_lock:
        return jsonify({"logs": experiment_state["logs"]})

if __name__ == '__main__':
    print("=" * 60)
    print("Time Constrained LLM Testing System - Web Interface")
    print("=" * 60)
    print("Starting server at http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)

