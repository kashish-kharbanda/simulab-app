# Agent Service API

The Agent Service provides a unified interface for interacting with AI agents in the application. It handles all the communication with the backend API endpoints.

## Core Features

### Task Creation

```typescript
createTask(payload: CreateTaskPayload): Promise<TaskResponse>
```

Creates a new task for an agent to process. This is the primary way to send requests to agents.

### Streaming Support

The Agent Service supports streaming responses from agents, allowing for real-time updates as the agent processes a request.

```typescript
connectToStream(
  taskId: string,
  onChunk: StreamEventCallback,
  onError?: ErrorCallback,
  onConnected?: ConnectionCallback,
  onCompleted?: CompletionCallback,
): () => void
```

This function returns a disconnect function that can be called to terminate the stream early.

```typescript
disconnectStream(taskId: string): boolean
```

Disconnects a specific stream by task ID.

```typescript
disconnectAllStreams(): void
```

Disconnects all active streams.

## Usage Examples

### Creating a basic task

```typescript
const response = await agentService.createTask({
  agent_name: "capybara",
  message: {
    type: "json",
    author: "user",
    style: "static",
    data: {
      query: "What's the weather like today?",
      document_id: `user_query_${Date.now()}`,
    },
  },
  require_approval: false,
});

console.log(response.task_id);
console.log(response.message?.data?.response);
```

### Using streaming

```typescript
const response = await agentService.createTask({
  agent_name: "capybara",
  message: {
    // ...message details
  }
});

if (response.task_id) {
  const disconnect = agentService.connectToStream(
    response.task_id,
    (chunk) => {
      // Handle each chunk of the stream
      console.log("Received chunk:", chunk.content_chunk);
      
      // Update UI with streaming content
      updateStreamingContent(chunk.content_chunk);
    },
    (error) => console.error("Stream error:", error),
    () => console.log("Stream connected successfully"),
    (taskId) => console.log("Stream completed for task:", taskId)
  );
  
  // Store the disconnect function to use later if needed
  // For example, clean up on component unmount:
  // useEffect(() => {
  //   return () => disconnect();
  // }, []);
}
```

## Type Definitions

The service exports several TypeScript interfaces to ensure type safety:

- `CreateTaskPayload`: Structure for creating a new task
- `TaskResponse`: Response structure from creating a task
- `Message` & `MessageData`: Structures for message content
- `TaskEvent`: Structure for streaming events
- Various callback types for streaming events 