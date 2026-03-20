import os
import sys
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

# Store experiment state (in-memory for serverless - will reset on cold starts)
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

def log_message(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    experiment_state["logs"].append({
        "timestamp": timestamp,
        "message": message
    })
    # Keep only last 100 logs
    if len(experiment_state["logs"]) > 100:
        experiment_state["logs"] = experiment_state["logs"][-100:]

def run_experiment_sync(config_dict):
    global experiment_state
    
    try:
        experiment_state["status"] = "running"
        experiment_state["start_time"] = time.time()
        experiment_state["logs"] = []
        experiment_state["error"] = None
        
        log_message(f"Starting experiment: {config_dict.get('experiment_name', 'default')}")
        
        # Create experiment config
        from run_pipeline import ExperimentConfig, ExperimentRunner
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
        
        experiment_state["total_tasks"] = config.task_count
        
        # Run experiment
        runner = ExperimentRunner(config)
        result = runner.run()
        
        # Store results
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
        
        return {"status": "completed", "message": "Experiment completed successfully"}
        
    except Exception as e:
        experiment_state["status"] = "error"
        experiment_state["end_time"] = time.time()
        experiment_state["error"] = str(e)
        log_message(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/experiment')
def experiment():
    return render_template('experiment.html')

@app.route('/api/config', methods=['GET'])
def get_default_config():
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
    
    result = run_experiment_sync(config)
    
    return jsonify(result)

@app.route('/api/status', methods=['GET'])
def get_status():
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
    if experiment_state["status"] == "running":
        experiment_state["status"] = "stopped"
        experiment_state["end_time"] = time.time()
        log_message("Experiment stopped by user")
        return jsonify({"status": "stopped", "message": "Experiment stopped"})
    else:
        return jsonify({"error": "No experiment running"}), 400

@app.route('/api/results', methods=['GET'])
def get_results():
    return jsonify({
        "results": experiment_state["results"],
        "statistics": experiment_state.get("statistics", {}),
        "status": experiment_state["status"]
    })

@app.route('/api/logs', methods=['GET'])
def get_logs():
    return jsonify({"logs": experiment_state["logs"]})

@app.route('/api/test-prompt', methods=['POST'])
def test_prompt():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "Prompt cannot be empty"}), 400
        
        time_limit = data.get("time_limit", 10.0)
        if time_limit <= 0:
            time_limit = 10.0
        
        # Import LLM manager
        from core.llm import get_llm_manager
        llm_manager = get_llm_manager()
        
        # Generate response with time limit
        response = llm_manager.generate_response(prompt, time_limit)
        
        return jsonify({
            "success": True,
            "prompt": prompt,
            "time_limit": time_limit,
            "response": {
                "content": response.content,
                "status": response.status.value if hasattr(response.status, 'value') else str(response.status),
                "time_elapsed": response.time_elapsed,
                "token_count": response.token_count,
                "model": response.model,
                "finish_reason": response.finish_reason
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development")
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Time Constrained LLM Testing System - Web Interface")
    print("=" * 60)
    print("Starting server at http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5001)
