-- employees
INSERT INTO employees (name, email, role) VALUES
 ('Youssef Adel', 'youssef.adel@swiftrail.com', 'sales_rep'),
 ('Mona Khalil', 'mona.khalil@swiftrail.com', 'sales_rep'),
 ('Sherif Nassar', 'sherif.nassar@swiftrail.com', 'finance_manager');

-- customers
INSERT INTO customers (name, credit_limit, balance_due, credit_status) VALUES
 ('Delta Textiles Co.', 500000, 12000, 'good'),
 ('Nile Grain Traders', 300000, 45000, 'hold'),
 ('Red Sea Steel Imports', 800000, 210000, 'hold'),
 ('Cairo Ceramics Ltd.', 250000, 8000, 'good');

-- shipments
INSERT INTO shipments (customer_id, origin, destination, railcar_id, base_rate, final_rate, status, requested_by) VALUES
 (1, 'Alexandria Port', 'Cairo Yard', 'RC-1042', 85000, 85000, 'pending', 1),
 (2, '10th of Ramadan', 'Sohag Yard', 'RC-2210', 62000, NULL, 'blocked', 2),
 (3, 'Ain Sokhna Port', 'Aswan Yard', 'RC-3387', 140000, NULL, 'blocked', 1),
 (4, 'Cairo Yard', 'Alexandria Port', 'RC-1188', 40000, 36000, 'released', 2),
 (1, 'Cairo Yard', 'Luxor Yard', 'RC-1509', 95000, NULL, 'pending', 1);
INSERT INTO shipments (
 customer_id, origin, destination, railcar_id, base_rate,
 final_rate, status, requested_by
) VALUES
 (1, 'Alexandria Port', 'Cairo Yard', 'RC-1660', 105000, 105000, 'delivery_exception', 1);

-- invoices
INSERT INTO invoices (customer_id, shipment_id, amount, due_date, paid_status, days_overdue) VALUES
 (1, NULL, 12000, '2026-06-15', 'paid', 0),
 (2, NULL, 45000, '2026-06-01', 'overdue', 30),
 (3, NULL, 130000, '2026-04-01', 'overdue', 95),
 (3, NULL, 80000, '2026-04-20', 'overdue', 91),
 (4, NULL, 8000, '2026-07-10', 'unpaid', 0);

-- credit_holds
INSERT INTO credit_holds (customer_id, reason, severity, status, released_by, released_at) VALUES
 (2, 'Invoice #2 30 days past due', 'minor', 'active', NULL, NULL),
 (3, 'Invoices #3/#4 more than 90 days past due, balance exceeds 25% of credit limit', 'severe', 'active', NULL, NULL),
 (4, 'Old invoice 45 days past due (resolved)', 'minor', 'released', 3, '2026-05-02 10:15:00');

-- rate_exceptions
INSERT INTO rate_exceptions (shipment_id, requested_by, discount_pct, justification, status, approved_by, resolved_at) VALUES
 (4, 2, 10, 'Long-term customer, matching a competitor quote for this lane; within rep authority.', 'auto_approved', NULL, '2026-05-20 09:00:00'),
 (5, 1, 25, 'Customer bundling three future shipments this quarter; requesting deeper discount to secure the volume commitment.', 'pending', NULL, NULL),
 (2, 2, 30, 'Requested to offset customer complaint about a delayed prior shipment.', 'rejected', 3, '2026-06-18 14:30:00');
