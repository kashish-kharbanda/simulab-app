/**
 * Type definitions for streaming functionality
 */

import { TaskMessageEntry } from './messages';

/**
 * Event type enum
 */
export enum EventType {
  FULL_CONTENT_UPDATE = "full_content_update",
  STREAMING_CHUNK = "streaming_chunk",
  DONE = "done",
  ERROR = "error",
  TOOL_REQUEST = "tool_request",
  TOOL_RESPONSE = "tool_response",
}

/**
 * Base interface for all task message stream events
 */
export interface BaseTaskEvent {
  parent_task_message_entry: TaskMessageEntry;
}

/**
 * Event for streaming chunks of content
 */
export interface StreamChunkEvent extends BaseTaskEvent {
  type: EventType.STREAMING_CHUNK;
  content_chunk: string;
}

/**
 * Event for indicating the task is done
 */
export interface DoneEvent extends BaseTaskEvent {
  type: EventType.DONE;
}

/**
 * Event for streaming the full content
 */
export interface FullContentEvent extends BaseTaskEvent {
  type: EventType.FULL_CONTENT_UPDATE;
  content: string;
}

/**
 * Event for streaming errors
 */
export interface ErrorEvent extends BaseTaskEvent {
  type: EventType.ERROR;
  error_message: string;
}

/**
 * Event for tool requests
 */
export interface ToolRequestEvent extends BaseTaskEvent {
  type: EventType.TOOL_REQUEST;
  name: string;
  arguments: Record<string, any>;
}

/**
 * Event for tool responses
 */
export interface ToolResponseEvent extends BaseTaskEvent {
  type: EventType.TOOL_RESPONSE;
  name: string;
  content: any;
}

/**
 * Union type for all possible task events
 */
export type TaskEvent = StreamChunkEvent | DoneEvent | FullContentEvent | ErrorEvent | ToolRequestEvent | ToolResponseEvent;

/**
 * Callback type for handling stream events/chunks
 */
export type StreamEventCallback = (data: TaskEvent) => void;

/**
 * Callback type for handling stream errors
 */
export type ErrorCallback = (error: Error | Event) => void;

/**
 * Callback type for handling stream connection events
 */
export type ConnectionCallback = () => void;

/**
 * Callback type for handling stream completion
 */
export type CompletionCallback = (taskId: string) => void;

/**
 * Interface representing an active stream connection
 */
export interface StreamConnection {
  eventSource: EventSource;
  disconnect: () => void;
} 