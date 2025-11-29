/**
 * TypeScript equivalent of the Python message types used in AgentEx
 */

/**
 * Message type enum
 */
export enum MessageType {
  TEXT = "text",
  JSON = "json",
  TOOL_REQUEST = "tool_request",
  TOOL_RESPONSE = "tool_response",
}

/**
 * Message author enum
 */
export enum MessageAuthor {
  USER = "user",
  AGENT = "agent",
}

/**
 * Message style enum
 */
export enum MessageStyle {
  STATIC = "static",
  ACTIVE = "active",
}

/**
 * Text format enum
 */
export enum TextFormat {
  MARKDOWN = "markdown",
  PLAIN = "plain",
  CODE = "code",
}

/**
 * File attachment interface
 */
export interface FileAttachment {
  /**
   * The unique ID of the attached file
   */
  file_id: string;
  
  /**
   * The name of the file
   */
  name: string;
  
  /**
   * The size of the file in bytes
   */
  size: number;
  
  /**
   * The MIME type or content type of the file
   */
  type: string;
  
  /**
   * The status of the file upload
   */
  status?: 'success' | 'error' | 'uploading';
  
  /**
   * The error message for failed uploads
   */
  error?: string;
  
  /**
   * The progress of the file upload
   */
  progress?: number;
}

/**
 * Base interface for all task messages
 */
export interface BaseMessage {
  /**
   * The type of the message
   */
  type: MessageType;
  
  /**
   * The role of the messages author
   */
  author: MessageAuthor;
  
  /**
   * The style of the message, used by clients to determine display
   */
  style: MessageStyle;
}

/**
 * Text message interface
 */
export interface TextMessage extends BaseMessage {
  /**
   * The specific message type, always 'text'
   */
  type: MessageType.TEXT;
  
  /**
   * The format of the message for display
   */
  format: TextFormat;
  
  /**
   * The contents of the text message
   */
  content: string;
  
  /**
   * Optional list of file attachments with metadata
   */
  attachments?: FileAttachment[];
}

/**
 * JSON message interface
 */
export interface JSONMessage extends BaseMessage {
  /**
   * The specific message type, always 'json'
   */
  type: MessageType.JSON;
  
  /**
   * The contents of the JSON message
   */
  data: Record<string, any>;
}

/**
 * Tool request message interface
 */
export interface ToolRequestMessage extends BaseMessage {
  /**
   * The specific message type, always 'tool_request'
   */
  type: MessageType.TOOL_REQUEST;
  
  /**
   * The name of the tool
   */
  name: string;
  
  /**
   * The arguments for the tool
   */
  arguments: Record<string, any>;
}

/**
 * Tool response message interface
 */
export interface ToolResponseMessage extends BaseMessage {
  /**
   * The specific message type, always 'tool_response'
   */
  type: MessageType.TOOL_RESPONSE;
  
  /**
   * The name of the tool
   */
  name: string;
  
  /**
   * The content of the tool response
   */
  content: any;
}

/**
 * Union type representing all possible task messages
 */
export type TaskMessage = TextMessage | JSONMessage | ToolRequestMessage | ToolResponseMessage;

/**
 * Message entry as stored in the database
 */
export interface TaskMessageEntry {
  /**
   * The unique ID of the message entry
   */
  id?: string;
  
  /**
   * ID of the task this message belongs to
   */
  task_id: string;
  
  /**
   * The actual message content
   */
  message: TaskMessage;
  
  /**
   * The timestamp when the message was created
   */
  timestamp: string;
  
  /**
   * Additional metadata for the message
   */
  metadata: Record<string, any>;
}

/**
 * Helper function to create a JSON message
 */
export function createJSONMessage(
  data: any,
  author: MessageAuthor = MessageAuthor.USER,
  style: MessageStyle = MessageStyle.STATIC
): JSONMessage {
  return {
    type: MessageType.JSON,
    author,
    style,
    data
  };
}

/**
 * Helper function to create a text message
 */
export function createTextMessage(
  content: string,
  format: TextFormat = TextFormat.PLAIN,
  author: MessageAuthor = MessageAuthor.USER,
  style: MessageStyle = MessageStyle.STATIC,
  attachments?: FileAttachment[]
): TextMessage {
  return {
    type: MessageType.TEXT,
    author,
    style,
    format,
    content,
    attachments
  };
} 