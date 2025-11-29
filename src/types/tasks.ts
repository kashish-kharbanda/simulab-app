import { TaskMessage } from "./messages";

// Enum for task status values
export enum TaskStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  ERROR = "error",
  CANCELLED = "cancelled"
}

// Interface for the task creation payload
export interface CreateTaskPayload {
  /**
   * The unique name of the agent to use to run the task
   */
  agent_name: string;

  /**
   * The user's message for the task
   */
  message: TaskMessage;

  /**
   * Whether the task requires human approval in order to complete.
   * If false, the task is left running until the human sends a finish
   */
  require_approval?: boolean;
}

// Interface for the task response
export interface CreateTaskResponse {
  id: string;
  agent_id: string;
}

// Interface for the Task model
export interface Task {
  /**
   * Unique Task ID
   */
  id: string;
  
  /**
   * The ID of the agent that is responsible for this task
   */
  agent_id: string;
  
  /**
   * The current status of the task
   */
  status?: TaskStatus;
  
  /**
   * The reason for the current task status
   */
  status_reason?: string;
}

// Interface for the TaskEntry model
export interface TaskEntry extends Task {
  /**
   * The timestamp when the task was created
   */
  created_at: Date;
  
  /**
   * The timestamp when the task was last updated
   */
  updated_at?: Date;
}

// Interface for error responses
export interface ErrorResponse {
  error: string;
}

// Union type for task responses
export type TaskResponse = CreateTaskResponse | ErrorResponse;

// Enum for task signal types
export enum TaskSignalType {
  INSTRUCT = "instruct",
  APPROVE = "approve",
  CANCEL = "cancel"
}

// Interface for the signal task payload
export interface BaseSignalTaskPayload {
  type: TaskSignalType;
}

export interface CancelTaskPayload extends BaseSignalTaskPayload {
  type: TaskSignalType.CANCEL;
}

export interface ApproveTaskPayload extends BaseSignalTaskPayload {
  type: TaskSignalType.APPROVE;
}

export interface InstructTaskPayload extends BaseSignalTaskPayload {
  type: TaskSignalType.INSTRUCT;
  message: TaskMessage;
}

// Union type for signal task requests with type discriminator
export type SignalTaskPayload = CancelTaskPayload | ApproveTaskPayload | InstructTaskPayload; 