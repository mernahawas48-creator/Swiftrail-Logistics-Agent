-- Apply once to an existing Swiftrail database created before the final project.
-- Run as a schema-owning MySQL account; the application account keeps CRUD-only
-- permissions during normal operation.

ALTER TABLE shipments
    MODIFY COLUMN status ENUM(
        'pending','blocked','released','in_transit','delivered',
        'delivery_exception'
    ) NOT NULL DEFAULT 'pending';

CREATE TABLE IF NOT EXISTS delivery_recovery_cases (
    id                    INT PRIMARY KEY AUTO_INCREMENT,
    shipment_id           INT NOT NULL,
    customer_id           INT NOT NULL,
    failure_reason        VARCHAR(500) NOT NULL,
    case_status           ENUM('open','waiting_customer','waiting_admin','resolved') NOT NULL DEFAULT 'open',
    selected_option       VARCHAR(50),
    requested_destination VARCHAR(100),
    estimated_cost        DECIMAL(12,2),
    requires_admin        BOOLEAN NOT NULL DEFAULT FALSE,
    idempotency_key       VARCHAR(180) UNIQUE,
    created_by            INT NOT NULL,
    applied_by            INT,
    created_at            TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at           TIMESTAMP NULL,
    FOREIGN KEY (shipment_id) REFERENCES shipments(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (created_by) REFERENCES employees(id),
    FOREIGN KEY (applied_by) REFERENCES employees(id)
);

CREATE TABLE IF NOT EXISTS graph_runs (
    run_id VARCHAR(64) PRIMARY KEY,
    graph_name VARCHAR(100) NOT NULL,
    status VARCHAR(32) NOT NULL,
    current_node VARCHAR(100) NOT NULL,
    revision INT NOT NULL DEFAULT 0,
    state_json JSON NOT NULL,
    created_at VARCHAR(40) NOT NULL,
    updated_at VARCHAR(40) NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_checkpoints (
    checkpoint_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id VARCHAR(64) NOT NULL,
    revision INT NOT NULL,
    node VARCHAR(100) NOT NULL,
    event VARCHAR(100) NOT NULL,
    state_json JSON NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_graph_checkpoint_revision (run_id, revision),
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE IF NOT EXISTS graph_node_executions (
    execution_key VARCHAR(180) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    node VARCHAR(100) NOT NULL,
    result_json JSON NOT NULL,
    completed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE IF NOT EXISTS graph_hitl_tasks (
    task_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    node VARCHAR(100) NOT NULL,
    status ENUM('pending','approved','rejected') NOT NULL,
    reason VARCHAR(500) NOT NULL,
    request_json JSON NOT NULL,
    state_json JSON NOT NULL,
    decision_json JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP NULL,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE IF NOT EXISTS graph_failure_tickets (
    ticket_id VARCHAR(64) PRIMARY KEY,
    run_id VARCHAR(64) NOT NULL,
    failed_node VARCHAR(100) NOT NULL,
    status ENUM('open','investigating','resolved') NOT NULL,
    error_type VARCHAR(150) NOT NULL,
    error_message VARCHAR(1000) NOT NULL,
    state_json JSON NOT NULL,
    resolution_note VARCHAR(1000),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    investigating_at TIMESTAMP NULL,
    resolved_at TIMESTAMP NULL,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

INSERT INTO shipments (
    customer_id, origin, destination, railcar_id, base_rate,
    final_rate, status, requested_by
)
SELECT 1, 'Alexandria Port', 'Cairo Yard', 'RC-1660', 105000,
       105000, 'delivery_exception', 1
WHERE NOT EXISTS (
    SELECT 1 FROM shipments WHERE railcar_id = 'RC-1660'
);
