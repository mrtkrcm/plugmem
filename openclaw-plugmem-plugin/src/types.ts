// ── Graph Management ────────────────────────────────────────────────

export interface GraphCreateRequest {
  graph_id?: string;
}

export interface GraphResponse {
  graph_id: string;
  stats: Record<string, number>;
}

export interface GraphListResponse {
  graphs: string[];
}

export interface StatsResponse {
  graph_id: string;
  stats: Record<string, number>;
}

export interface NodeListResponse {
  graph_id: string;
  node_type: string;
  count: number;
  nodes: Record<string, unknown>[];
}

// ── Memory Insertion ────────────────────────────────────────────────

export interface TrajectoryStep {
  observation: string;
  action: string;
}

export interface SemanticMemoryInput {
  semantic_memory: string;
  tags?: string[];
}

export interface ProceduralMemoryInput {
  subgoal: string;
  procedural_memory: string;
  return?: number;
}

export interface EpisodicStep {
  observation?: string;
  action?: string;
  subgoal?: string;
  state?: string;
  reward?: string;
  time?: string | number;
}

export interface TrajectoryInsertRequest {
  mode: "trajectory";
  goal: string;
  steps: TrajectoryStep[];
  /**
   * Stamps every node created by this insert with the given session id.
   * Used by PlugMem's Sessions view + recall audit log.
   */
  session_id?: string;
}

export interface StructuredInsertRequest {
  mode: "structured";
  episodic?: EpisodicStep[][];
  semantic?: SemanticMemoryInput[];
  procedural?: ProceduralMemoryInput[];
  session_id?: string;
}

export type MemoryInsertRequest =
  | TrajectoryInsertRequest
  | StructuredInsertRequest;

export interface MemoryInsertResponse {
  status: string;
  stats: Record<string, number>;
}

// ── Retrieval & Reasoning ───────────────────────────────────────────

export interface RetrieveRequest {
  observation: string;
  goal?: string;
  subgoal?: string;
  state?: string;
  task_type?: string;
  time?: string;
  mode?: "semantic_memory" | "episodic_memory" | "procedural_memory" | null;
  source_in?: string[];
  min_confidence?: number;
  provenance_filters?: Record<string, string[]>;
}

export interface RetrieveResponse {
  mode: string;
  reasoning_prompt: Array<{ role: string; content: string }>;
  variables: Record<string, unknown>;
}

export interface ReasonRequest {
  observation: string;
  goal?: string;
  subgoal?: string;
  state?: string;
  task_type?: string;
  time?: string;
  mode?: "semantic_memory" | "episodic_memory" | "procedural_memory" | null;
  source_in?: string[];
  min_confidence?: number;
  provenance_filters?: Record<string, string[]>;
}

export interface ReasonResponse {
  mode: string;
  reasoning: string;
  reasoning_prompt: Array<{ role: string; content: string }>;
}

// ── Promotion ───────────────────────────────────────────────────────

export interface CandidateInput {
  kind: string;
  window: string;
}

export interface PromoteRequest {
  candidates: CandidateInput[];
  source_in?: string[];
  min_confidence?: number;
}

export interface PromotedMemory {
  node_type: string;
  node_id: number;
  memory: Record<string, unknown>;
}

export interface RejectedCandidate {
  index: number;
  kind: string;
  reason: string;
}

export interface PromoteResponse {
  inserted: PromotedMemory[];
  dropped: RejectedCandidate[];
}

// ── Consolidation ───────────────────────────────────────────────────

export interface ConsolidateRequest {
  merge_threshold?: number;
  max_merges_per_node?: number;
  max_candidates_per_tag?: number;
  max_total_candidates?: number;
  min_credibility_to_keep_active?: number;
  credibility_decay?: number;
  only_update_recent_window?: number | null;
  allow_merge_with_common_episodic_nodes?: boolean;
}

export interface ConsolidateResponse {
  status: string;
  stats: Record<string, number>;
}

// ── Health ───────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
  llm_available: boolean;
  embedding_available: boolean;
  chroma_available: boolean;
}

// ── Client errors ────────────────────────────────────────────────────

export class PlugMemError extends Error {
  constructor(
    message: string,
    public readonly statusCode: number,
    public readonly body?: unknown,
  ) {
    super(message);
    this.name = "PlugMemError";
  }
}

export class PlugMemConnectionError extends Error {
  constructor(
    message: string,
    public readonly cause?: unknown,
  ) {
    super(message);
    this.name = "PlugMemConnectionError";
  }
}
