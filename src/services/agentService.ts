// Import event types
import {
  StreamEventCallback,
  ErrorCallback,
  ConnectionCallback,
  CompletionCallback,
  StreamConnection,
  EventType,
  StreamChunkEvent,
  DoneEvent,
  FullContentEvent,
  ErrorEvent,
  ToolRequestEvent,
  ToolResponseEvent,
} from '@/types/events';

// Import message types
import {
  TaskMessage,
  TaskMessageEntry,
  MessageAuthor,
  MessageStyle,
  TextFormat,
  MessageType,
} from '@/types/messages';

// Import task types
import {
  CreateTaskPayload,
  CreateTaskResponse,
  TaskResponse,
  TaskSignalType,
  SignalTaskPayload,
  Task,
} from '@/types/tasks';

// Map to keep track of active streams by taskId
const activeStreams = new Map<string, StreamConnection>();

const inferEnvOrigin = () => {
  const port = process.env.PORT || '3000';
  const host = process.env.HOST || 'localhost';

  return (
    process.env.NEXT_PUBLIC_APP_URL ||
    process.env.NEXT_PUBLIC_SITE_URL ||
    (process.env.NEXT_PUBLIC_VERCEL_URL
      ? `https://${process.env.NEXT_PUBLIC_VERCEL_URL}`
      : undefined) ||
    process.env.APP_URL ||
    (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : undefined) ||
    process.env.SITE_URL ||
    `http://${host}:${port}`
  );
};

/**
 * Ensure API requests resolve to an absolute URL regardless of whether the code
 * executes on the server or the browser. When running in the browser we prefer
 * the current window origin (so cookies stay in scope); however, if the app was
 * opened from a `file://` path we fall back to the configured env origin.
 */
const resolveApiUrl = (path: string) => {
  const envOrigin = inferEnvOrigin();

  if (typeof window !== 'undefined') {
    const origin =
      (window.location && /^https?:/i.test(window.location.protocol)
        ? window.location.origin
        : undefined) || envOrigin;
    return new URL(path, origin).toString();
  }

  return new URL(path, envOrigin).toString();
};

const apiFetch = (path: string, init?: RequestInit) => fetch(resolveApiUrl(path), init);

// Define the agent service interface
interface AgentService {
  createTask(payload: CreateTaskPayload): Promise<CreateTaskResponse>;
  signalTask(taskId: string, payload: SignalTaskPayload): Promise<TaskResponse>;
  sendFollowUpMessage(taskId: string, message: TaskMessage): Promise<TaskResponse>;
  approveTask(taskId: string): Promise<TaskResponse>;
  cancelTask(taskId: string): Promise<TaskResponse>;
  connectToStream(
    taskId: string,
    onChunk: StreamEventCallback,
    onError?: ErrorCallback,
    onConnected?: ConnectionCallback,
    onCompleted?: CompletionCallback
  ): () => void;
  disconnectAllStreams(): void;
  disconnectStream(taskId: string): boolean;
  hasActiveStream(taskId: string): boolean;
  getTask(taskId: string): Promise<TaskResponse>;
  getTaskSpans(taskId: string, traceId: string): Promise<any[]>;
  processEventData(data: string, callback: StreamEventCallback, taskId?: string): void;
  getTaskMessages(taskId: string): Promise<TaskMessageEntry[]>;
  sendSignal(taskId: string, payload: SignalTaskPayload): Promise<TaskResponse>;
  listTasks(): Promise<Task[]>;
  listMessages(taskId: string, limit?: number): Promise<TaskMessageEntry[]>;
}

/**
 * Agent service for interacting with the backend
 */
export const agentService: AgentService = {
  /**
   * Creates a new task for an agent
   * 
   * @param payload - The task creation payload
   * @returns Promise with task response
   */
  async createTask(payload: CreateTaskPayload): Promise<CreateTaskResponse> {
    const response = await apiFetch('/api/tasks/create', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error('Failed to create task');
    }

    return response.json();
  },
  
  /**
   * Sends a signal to an existing task
   * @param taskId The ID of the task to signal
   * @param payload The signal task payload
   * @returns Promise with task response
   */
  async signalTask(taskId: string, payload: SignalTaskPayload): Promise<TaskResponse> {
    const response = await apiFetch(`/api/tasks/${taskId}/signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      return { error: text || 'Failed to send signal' };
    }
    return response.json();
  },
  
  /**
   * Sends a follow-up instruction to an existing task
   * @param taskId The ID of the task to signal
   * @param message The message to send
   * @returns Promise with task response
   */
  async sendFollowUpMessage(
    taskId: string,
    message: TaskMessage
  ): Promise<TaskResponse> {
    return this.signalTask(taskId, {
      type: TaskSignalType.INSTRUCT,
      message
    }); 
  },
  
  /**
   * Approves a task that requires approval
   * @param taskId The ID of the task to approve
   * @returns Promise with task response
   */
  async approveTask(taskId: string): Promise<TaskResponse> {
    return this.signalTask(taskId, {
      type: TaskSignalType.APPROVE
    });
  },
  
  /**
   * Cancels a running task
   * @param taskId The ID of the task to cancel
   * @returns Promise with task response
   */
  async cancelTask(taskId: string): Promise<TaskResponse> {
    return this.signalTask(taskId, {
      type: TaskSignalType.CANCEL
    });
  },
  
  /**
   * Connect to an SSE stream for a specific taskId
   * @param taskId The task ID to connect to
   * @param onChunk Callback for each chunk of streamed data
   * @param onError Callback for stream errors
   * @param onConnected Callback when connection is established
   * @param onCompleted Callback when stream is completed
   * @returns A function to disconnect from the stream
   */
  connectToStream(
    taskId: string,
    onChunk: StreamEventCallback,
    onError?: ErrorCallback,
    onConnected?: ConnectionCallback,
    onCompleted?: CompletionCallback,
  ): () => void {
    // If we already have a connection for this taskId, close it
    if (activeStreams.has(taskId)) {
      console.log(`Closing existing stream for task ${taskId} before creating a new one`);
      activeStreams.get(taskId)?.disconnect();
      activeStreams.delete(taskId);
    }

    console.log(`Connecting to SSE stream for task: ${taskId}`);

    try {
      // Connect to the stream
      const eventSource = new EventSource(`/api/tasks/${taskId}/stream`);

      // Handle general messages
      eventSource.onmessage = event => {
        try {
          // Log the raw event data for debugging

          // Check for completion message using the SSE format conventions
          let isCompletionEvent = false;

          // Handle case where we have data: prefix (standard SSE format)
          if (event.data.includes('data: ')) {
            const chunks = event.data.split('data: ');

            // Look for completion in any of the chunks
            for (const chunk of chunks) {
              if (!chunk.trim()) continue;
              const parsedChunk = JSON.parse(chunk.trim());
              if (parsedChunk && parsedChunk.type === 'done') {
                isCompletionEvent = true;
                break;
              }
            }
          } else {
            // Direct data without prefix (might happen in some implementations)
            try {
              const parsedData = JSON.parse(event.data);
              if (parsedData && parsedData.type === 'done') {
                isCompletionEvent = true;
              }
            } catch (e) {
              // Not JSON or not our format, check for legacy format
              if (event.data.includes('<done>')) {
                isCompletionEvent = true;
              }
            }
          }

          // Handle completion event if detected
          if (isCompletionEvent) {
            console.log(`Received completion signal for task: ${taskId}`);

            // Close the connection
            eventSource.close();
            activeStreams.delete(taskId);

            // Call the completion callback if provided
            if (onCompleted) {
              onCompleted(taskId);
            }

            return;
          }

          // Parse data and handle multiple chunks if needed
          this.processEventData(event.data, onChunk, taskId);
        } catch (error) {
          console.error('Error processing SSE data:', error);
        }
      };

      // Handle connection open
      eventSource.onopen = () => {
        console.log(`SSE connection opened for task: ${taskId}`);
        if (onConnected) {
          onConnected();
        }
      };

      // Handle errors
      eventSource.onerror = error => {
        console.error(`SSE error for task ${taskId}:`, error);

        if (onError) {
          onError(error);
        }

        // If the connection is closed, clean up
        if (eventSource.readyState === EventSource.CLOSED) {
          console.log(`SSE connection closed for task: ${taskId}`);
          activeStreams.delete(taskId);
        }
      };

      // Create the disconnect function
      const disconnect = () => {
        console.log(`Manually disconnecting SSE for task: ${taskId}`);
        eventSource.close();
        activeStreams.delete(taskId);
      };

      // Store the active connection
      activeStreams.set(taskId, { eventSource, disconnect });

      // Return the disconnect function
      return disconnect;
    } catch (err) {
      console.error(`Failed to connect to SSE for task ${taskId}:`, err);
      if (onError && err instanceof Error) {
        onError(err);
      }

      // Return a no-op function if connection failed
      return () => {};
    }
  },

  /**
   * Disconnect from all active streams
   */
  disconnectAllStreams(): void {
    console.log(`Disconnecting all ${activeStreams.size} active streams`);

    activeStreams.forEach((connection, taskId) => {
      console.log(`Closing stream for task: ${taskId}`);
      connection.disconnect();
    });

    activeStreams.clear();
  },

  /**
   * Disconnect from a specific stream by taskId
   * @param taskId The task ID to disconnect from
   * @returns true if the stream was found and disconnected, false otherwise
   */
  disconnectStream(taskId: string): boolean {
    if (activeStreams.has(taskId)) {
      const connection = activeStreams.get(taskId);
      connection?.disconnect();
      activeStreams.delete(taskId);
      return true;
    }

    return false;
  },
  
  /**
   * Check if there is an active stream for a given taskId
   * @param taskId The task ID to check
   * @returns true if there is an active stream, false otherwise
   */
  hasActiveStream(taskId: string): boolean {
    return activeStreams.has(taskId);
  },
  
  /**
   * Process event data from the SSE stream
   * @param data The raw data from the SSE event
   * @param callback Callback function to call with processed data
   * @param taskId Optional taskId to include with the callback
   * @private
   */
  processEventData(data: string, callback: StreamEventCallback, taskId?: string): void {
    console.log(`Processing event data:`, data.substring(0, 100) + (data.length > 100 ? '...' : ''));

    // Handle potential common formats for SSE data

    // 1. First attempt to understand if this is SSE data or direct message payload
    if (data.includes('data: ')) {
      console.log('Data contains SSE format with "data: " prefix');
      // Split the data into chunks and process each one
      const chunks = data.split('data: ');
      console.log(`Split into ${chunks.length} chunks`);

      // Process each chunk (skip the first empty chunk if it exists)
      chunks.forEach((chunk: string, index: number) => {
        if (!chunk.trim()) {
          console.log(`Chunk ${index} is empty, skipping`);
          return;
        }

        console.log(
          `Processing chunk ${index}:`,
          chunk.substring(0, 50) + (chunk.length > 50 ? '...' : ''),
        );

        // Try to handle the chunk - first try parsing as JSON
        try {
          const parsedData = JSON.parse(chunk);
          console.log(
            `Successfully parsed chunk ${index} as JSON:`,
            parsedData.type || 'no type field',
          );

          if (parsedData.type === EventType.STREAMING_CHUNK && parsedData.content_chunk) {
            // Standard streaming chunk format
            console.log(
              `Valid streaming_chunk with content:`,
              parsedData.content_chunk?.substring(0, 30) +
                (parsedData.content_chunk && parsedData.content_chunk.length > 30 ? '...' : ''),
            );
            
            // Ensure we have a valid StreamChunkEvent
            if (parsedData.parent_task_message_entry) {
              callback(parsedData as StreamChunkEvent);
            } else {
              console.warn('Received streaming_chunk without parent_task_message_entry:', parsedData);
            }
          } else if (parsedData.type === 'connected') {
            console.log('Connected event, skipping');
            // Just a connection message, no need to process
          } else if (parsedData.parent_task_message_entry) {
            // We have a parent message entry, try to determine the event type
            if (parsedData.type === EventType.FULL_CONTENT_UPDATE && parsedData.content) {
              callback(parsedData as FullContentEvent);
            } else if (parsedData.type === EventType.DONE) {
              callback(parsedData as DoneEvent);
            } else if (parsedData.type === EventType.ERROR && parsedData.error_message) {
              callback(parsedData as ErrorEvent);
            } else if (parsedData.type === EventType.TOOL_REQUEST && parsedData.name && parsedData.arguments) {
              callback(parsedData as ToolRequestEvent);
            } else if (parsedData.type === EventType.TOOL_RESPONSE && parsedData.name && parsedData.content) {
              callback(parsedData as ToolResponseEvent);
            } else if (parsedData.content_chunk) {
              // Create a StreamChunkEvent from the data
              const streamEvent: StreamChunkEvent = {
                type: EventType.STREAMING_CHUNK,
                content_chunk: parsedData.content_chunk,
                parent_task_message_entry: parsedData.parent_task_message_entry
              };
              callback(streamEvent);
            } else if (parsedData.content) {
              // Create a FullContentEvent from the data
              const fullContentEvent: FullContentEvent = {
                type: EventType.FULL_CONTENT_UPDATE,
                content: parsedData.content,
                parent_task_message_entry: parsedData.parent_task_message_entry
              };
              callback(fullContentEvent);
            } else {
              console.log(`Unrecognized JSON structure with parent_task_message_entry:`, parsedData);
            }
          } else if (parsedData.content) {
            // Legacy format with direct content field
            console.log('Found legacy content field');
            // We can't create a proper event without parent_task_message_entry
            console.warn('Received content without parent_task_message_entry:', parsedData);
          } else {
            console.warn(`Unrecognized JSON structure:`, parsedData);
          }
        } catch (e) {
          console.log(`Failed to parse chunk ${index} as JSON, treating as raw string`);
          // If it can't be parsed as JSON, we can't create a proper event
          console.warn('Received raw string data that could not be parsed as JSON:', chunk.trim());
        }
      });
    } else {
      // Try parsing as JSON first
      try {
        const parsedData = JSON.parse(data);

        if (parsedData.type === EventType.STREAMING_CHUNK && parsedData.content_chunk && parsedData.parent_task_message_entry) {
          console.log(
            `Valid streaming_chunk with content:`,
            parsedData.content_chunk?.substring(0, 30) +
              (parsedData.content_chunk && parsedData.content_chunk.length > 30 ? '...' : ''),
          );
          callback(parsedData as StreamChunkEvent);
        } else if (parsedData.type === EventType.FULL_CONTENT_UPDATE && parsedData.content && parsedData.parent_task_message_entry) {
          callback(parsedData as FullContentEvent);
        } else if (parsedData.type === EventType.DONE && parsedData.parent_task_message_entry) {
          callback(parsedData as DoneEvent);
        } else if (parsedData.type === EventType.ERROR && parsedData.error_message && parsedData.parent_task_message_entry) {
          callback(parsedData as ErrorEvent);
        } else if (parsedData.type === EventType.TOOL_REQUEST && parsedData.name && parsedData.arguments && parsedData.parent_task_message_entry) {
          callback(parsedData as ToolRequestEvent);
        } else if (parsedData.type === EventType.TOOL_RESPONSE && parsedData.name && parsedData.content && parsedData.parent_task_message_entry) {
          callback(parsedData as ToolResponseEvent);
        } else if (parsedData.parent_task_message_entry) {
          // We have a parent message entry but not a recognized event type
          console.log('Found parent_task_message_entry but unrecognized event type:', parsedData);
        } else {
          console.log(`Unrecognized JSON structure:`, parsedData);
        }
      } catch (e) {
        console.log(`Failed to parse as JSON, treating as raw string`);
        // If not JSON, we can't create a proper event
        console.warn('Received raw string data that could not be parsed as JSON:', data.trim());
      }
    }
  },
  
  /**
   * Creates a document-based task with file attachments
   * (Placeholder for future implementation)
   */
  // async createDocumentTask() {...}
  
  /**
   * Gets the status of a task
   * (Placeholder for future implementation)
   */
  // async getTaskStatus(taskId: string) {...}

  /**
   * Gets a task by ID
   * @param taskId The ID of the task to get
   * @returns Promise with task response
   */
  async getTask(taskId: string): Promise<TaskResponse> {
    const response = await fetch(`/api/tasks/${taskId}`);
    
    if (!response.ok) {
      throw new Error('Failed to fetch task');
    }
    
    return response.json();
  },

  /**
   * Gets all messages for a task
   * @param taskId The ID of the task to get messages for
   * @returns Promise with array of task message entries
   */
  async getTaskMessages(taskId: string): Promise<TaskMessageEntry[]> {
    const response = await apiFetch(`/api/tasks/${taskId}/messages`);

    if (!response.ok) {
      throw new Error('Failed to fetch task messages');
    }

    return response.json();
  },

  async sendSignal(taskId: string, signal: SignalTaskPayload): Promise<TaskResponse> {
    const response = await apiFetch(`/api/tasks/${taskId}/signal`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(signal),
    });

    if (!response.ok) {
      // Don't throw - return an ErrorResponse so callers can proceed gracefully
      const text = await response.text().catch(() => '');
      return { error: text || 'Failed to send signal' };
    }
    return response.json();
  },

  /**
   * Lists all tasks
   * @returns Promise with array of tasks
   */
  async listTasks(): Promise<Task[]> {
    try {
      const response = await apiFetch('/api/tasks');
      
      if (!response.ok) {
        // Backend unavailable - return empty array instead of throwing
        console.warn('[agentService] Backend unavailable for listTasks, returning empty array');
        return [];
      }
      
      return response.json();
    } catch (error) {
      // Network error - backend not running
      console.warn('[agentService] Could not reach backend for listTasks:', error);
      return [];
    }
  },

  /**
   * Lists all messages
   * @param taskId The ID of the task to get messages for
   * @param limit Optional limit on the number of messages to return
   * @returns Promise with array of task message entries
   */
  async listMessages(taskId: string, limit?: number): Promise<TaskMessageEntry[]> {
    const url = limit !== undefined ? `/api/tasks/${taskId}/messages?limit=${limit}` : `/api/tasks/${taskId}/messages`;
    const response = await apiFetch(url);

    if (!response.ok) {
      throw new Error('Failed to list messages');
    }

    const raw = await response.json();
    // Normalize to TaskMessageEntry[] shape if backend returns a different structure
    if (Array.isArray(raw)) {
      return raw.map((item: any) => {
        // If already in expected shape
        if (item?.message && (item.message.content !== undefined || item.message.data !== undefined)) {
          return item as TaskMessageEntry;
        }
        // If AgentEx backend returns { id, content: { author, content, format }, created_at, ... }
        const contentObj = item?.content;
        const author =
          contentObj?.author === 'agent' ? MessageAuthor.AGENT
          : contentObj?.author === 'user' ? MessageAuthor.USER
          : MessageAuthor.AGENT;
        const format =
          contentObj?.format === 'markdown' ? TextFormat.MARKDOWN
          : contentObj?.format === 'code' ? TextFormat.CODE
          : TextFormat.PLAIN;
        const textContent =
          typeof contentObj?.content === 'string'
            ? contentObj.content
            : JSON.stringify(contentObj?.content ?? '');
        const normalized: TaskMessageEntry = {
          id: item?.id,
          task_id: item?.task_id ?? taskId,
          message: {
            type: MessageType.TEXT,
            author,
            style: MessageStyle.STATIC,
            format,
            content: textContent,
          },
          timestamp: item?.created_at ?? item?.timestamp ?? new Date().toISOString(),
          metadata: item?.metadata ?? {},
        };
        return normalized;
      });
    }
    return raw;
  },

  /**
   * Fetches spans for a specific task and trace ID
   * @param taskId The ID of the task to get spans for
   * @param traceId The trace ID to filter spans by
   * @returns Promise with array of spans
   */
  async getTaskSpans(taskId: string, traceId: string): Promise<any[]> {
    // Add a timestamp to the URL to prevent caching
    const timestamp = new Date().getTime();
    
    const response = await apiFetch(`/api/tasks/${taskId}/spans?trace_id=${traceId}&_t=${timestamp}`, {
      // Add cache control headers to prevent browser caching
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate',
        Pragma: 'no-cache',
        Expires: '0',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch spans: ${response.status}`);
    }

    const spansData = await response.json();

    if (!Array.isArray(spansData)) {
      throw new Error('Invalid response format: expected an array of spans');
    }

    // Process the spans data to ensure dates are Date objects
    return spansData.map((span: any) => ({
      ...span,
      start_time: new Date(span.start_time),
      end_time: span.end_time ? new Date(span.end_time) : null,
    }));
  },
}; 