CREATE TABLE employees (
    id     INT PRIMARY KEY AUTO_INCREMENT,
    name   VARCHAR(100) NOT NULL,
    email  VARCHAR(150) NOT NULL UNIQUE,
    role   ENUM('sales_rep', 'finance_manager') NOT NULL
);

CREATE TABLE customers (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    name          VARCHAR(150) NOT NULL,
    credit_limit  DECIMAL(12,2) NOT NULL CHECK (credit_limit >= 0),
    balance_due   DECIMAL(12,2) NOT NULL DEFAULT 0 CHECK (balance_due >= 0),
    credit_status ENUM('good', 'hold') NOT NULL DEFAULT 'good'
);

CREATE TABLE shipments (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    customer_id   INT NOT NULL,
    origin        VARCHAR(100) NOT NULL,
    destination   VARCHAR(100) NOT NULL,
    railcar_id    VARCHAR(50),
    base_rate     DECIMAL(12,2) NOT NULL CHECK (base_rate > 0),
    final_rate    DECIMAL(12,2),
    status        ENUM('pending','blocked','released','in_transit','delivered','delivery_exception') NOT NULL DEFAULT 'pending',
    requested_by  INT NOT NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (requested_by) REFERENCES employees(id)
);

CREATE TABLE invoices (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    customer_id   INT NOT NULL,
    shipment_id   INT UNIQUE,
    amount        DECIMAL(12,2) NOT NULL CHECK (amount > 0),
    due_date      DATE NOT NULL,
    paid_status   ENUM('unpaid','paid','overdue') NOT NULL DEFAULT 'unpaid',
    days_overdue  INT NOT NULL DEFAULT 0 CHECK (days_overdue >= 0),
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (shipment_id) REFERENCES shipments(id)
);

CREATE TABLE credit_holds (
    id            INT PRIMARY KEY AUTO_INCREMENT,
    customer_id   INT NOT NULL,
    reason        VARCHAR(255) NOT NULL,
    severity      ENUM('minor','severe') NOT NULL,
    status        ENUM('active','released') NOT NULL DEFAULT 'active',
    placed_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    released_by   INT,
    released_at   TIMESTAMP NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (released_by) REFERENCES employees(id)
);

CREATE TABLE rate_exceptions (
    id             INT PRIMARY KEY AUTO_INCREMENT,
    shipment_id    INT NOT NULL,
    requested_by   INT NOT NULL,
    discount_pct   DECIMAL(5,2) NOT NULL CHECK (discount_pct > 0 AND discount_pct <= 50),
    justification  TEXT NOT NULL CHECK (CHAR_LENGTH(justification) >= 20),
    status         ENUM('pending','auto_approved','approved','rejected') NOT NULL DEFAULT 'pending',
    approved_by    INT,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at    TIMESTAMP NULL,
    FOREIGN KEY (shipment_id) REFERENCES shipments(id),
    FOREIGN KEY (requested_by) REFERENCES employees(id),
    FOREIGN KEY (approved_by) REFERENCES employees(id)
);

CREATE TABLE delivery_recovery_cases (
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

-- Shared durable state-graph runtime. HITL tasks and failure tickets are
-- intentionally separate so an expected human decision can never be confused
-- with an unplanned execution failure.
CREATE TABLE graph_runs (
    run_id        VARCHAR(64) PRIMARY KEY,
    graph_name    VARCHAR(100) NOT NULL,
    status        VARCHAR(32) NOT NULL,
    current_node  VARCHAR(100) NOT NULL,
    revision      INT NOT NULL DEFAULT 0,
    state_json    JSON NOT NULL,
    created_at    VARCHAR(40) NOT NULL,
    updated_at    VARCHAR(40) NOT NULL
);

CREATE TABLE graph_checkpoints (
    checkpoint_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id         VARCHAR(64) NOT NULL,
    revision       INT NOT NULL,
    node           VARCHAR(100) NOT NULL,
    event          VARCHAR(100) NOT NULL,
    state_json     JSON NOT NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_graph_checkpoint_revision (run_id, revision),
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE graph_node_executions (
    execution_key VARCHAR(180) PRIMARY KEY,
    run_id        VARCHAR(64) NOT NULL,
    node          VARCHAR(100) NOT NULL,
    result_json   JSON NOT NULL,
    completed_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE graph_hitl_tasks (
    task_id       VARCHAR(64) PRIMARY KEY,
    run_id        VARCHAR(64) NOT NULL,
    node          VARCHAR(100) NOT NULL,
    status        ENUM('pending','approved','rejected') NOT NULL,
    reason        VARCHAR(500) NOT NULL,
    request_json  JSON NOT NULL,
    state_json    JSON NOT NULL,
    decision_json JSON,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    resolved_at   TIMESTAMP NULL,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);

CREATE TABLE graph_failure_tickets (
    ticket_id        VARCHAR(64) PRIMARY KEY,
    run_id           VARCHAR(64) NOT NULL,
    failed_node      VARCHAR(100) NOT NULL,
    status           ENUM('open','investigating','resolved') NOT NULL,
    error_type       VARCHAR(150) NOT NULL,
    error_message    VARCHAR(1000) NOT NULL,
    state_json       JSON NOT NULL,
    resolution_note  VARCHAR(1000),
    created_at       TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    investigating_at TIMESTAMP NULL,
    resolved_at      TIMESTAMP NULL,
    FOREIGN KEY (run_id) REFERENCES graph_runs(run_id)
);
