# Delivery Exception Recovery and Rerouting Policy

## DR-1 — Customer Response

A delivery-exception recovery run must preserve the shipment state and wait for
the customer's explicit redelivery or rerouting choice. The agent must not
invent a customer decision or treat silence as approval.

## DR-2 — Automatic Redelivery

Redelivery to the existing verified destination may proceed without admin
approval when it introduces no customs change and no additional rerouting cost.

## DR-3 — Admin Review

Admin approval is required when the destination is not verified, the estimated
rerouting cost exceeds $500, the reroute changes the customs region, or the
shipment is high-value. The decision and rationale must be persisted before the
write tool runs.

## DR-4 — Revision Cycle

If the customer rejects the available options or an admin rejects a risky
reroute, the agent must generate revised options and wait for a new customer
choice rather than silently applying the rejected action.

## DR-5 — Safe Recovery

Unexpected MCP, database, schema-validation, retrieval, or model failures must
open a failure ticket. After resolution, execution resumes from the failed node
using the latest durable checkpoint and idempotent write key.
