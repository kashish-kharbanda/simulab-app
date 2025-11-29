/**
 * TypeScript equivalent of the Python agent types used in AgentEx
 */

/**
 * Interface for credential mapping
 */
export interface CredentialMapping {
  // TODO: Define credential mapping structure based on Python implementation
  [key: string]: any;
}

/**
 * Interface for creating a new agent
 */
export interface CreateAgentRequest {
  /**
   * The unique name of the agent
   */
  name: string;

  /**
   * The description of the agent
   */
  description: string;

  /**
   * The name of the workflow that defines the agent
   */
  workflow_name: string;

  /**
   * The name of the queue to send tasks to
   */
  workflow_queue_name: string;

  /**
   * The docker image to use for the agent.
   * Use this if you built the agent locally and pushed it to a registry.
   */
  docker_image?: string;

  /**
   * Optional list of credential mappings
   */
  credentials?: CredentialMapping[];

  /**
   * Environment variables to set directly in the agent deployment
   */
  env_vars?: Record<string, string>;
}

/**
 * Enum for agent status
 */
export enum AgentStatus {
  PENDING = "Pending",
  BUILDING = "Building",
  READY = "Ready",
  FAILED = "Failed",
  UNKNOWN = "Unknown"
}

/**
 * Interface for agent model
 */
export interface AgentModel {
  /**
   * The unique identifier of the agent
   */
  id: string;

  /**
   * The unique name of the agent
   */
  name: string;

  /**
   * The description of the action
   */
  description: string;

  /**
   * The status of the action, indicating if it's building, ready, failed, etc.
   */
  status: AgentStatus;

  /**
   * The reason for the status of the action
   */
  status_reason?: string;
} 