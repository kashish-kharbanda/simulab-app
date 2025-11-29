'use client';

import { useState, useRef, useEffect, useCallback, use } from 'react';
import { AnimatePresence } from 'framer-motion';
import { useToast } from '@/hooks/use-toast';
import { useTasks } from '@/context/TasksContext';
import { ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { Button } from '@/components/ui/button';
import { ArrowDown } from 'lucide-react';

import MessageCard from '@/components/chat/message-card';
import TaskControls from '@/components/chat/task-controls';
import TaskBadges from '@/components/chat/task-badges';
import { agentService } from '@/services/agentService';
import { FileAttachment } from '@/types/messages';
import { 
  TaskEvent, 
  StreamEventCallback, 
  ErrorCallback, 
  ConnectionCallback, 
  CompletionCallback,
  EventType,
  ToolRequestEvent,
  ToolResponseEvent,
} from '@/types/events';
import { 
  TaskMessageEntry, 
  MessageAuthor, 
  MessageStyle, 
  TextFormat, 
  createTextMessage, 
  MessageType, 
  TextMessage, 
  JSONMessage, 
  ToolRequestMessage, 
  ToolResponseMessage 
} from '@/types/messages';
import { TaskSignalType, Task, TaskStatus } from '@/types/tasks';
import ToolMessageCard from '@/components/chat/tool-message-card';

interface PageProps {
  params: Promise<{
    taskId: string;
  }>;
}

// Main Page component
const Page: React.FC<PageProps> = ({ params }) => {
  const { taskId } = use(params);
  const { toast } = useToast();
  const { refreshTasks, setSelectedTask } = useTasks();

  const [messageEntries, setMessageEntries] = useState<Record<string, TaskMessageEntry>>({});
  // State variables
  const [isLoading, setIsLoading] = useState(true);
  const [isThinking, setIsThinking] = useState(false);
  const [userInput, setUserInput] = useState<string>('');
  const [task, setTask] = useState<Task | null>(null);
  const [isApproving, setIsApproving] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isCopied, setIsCopied] = useState(false);
  const [autoScrollEnabled, setAutoScrollEnabled] = useState(true);
  const previousMessagesLength = useRef<number>(0);
  const eventSourceRef = useRef<(() => void) | null>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const newestMessageRef = useRef<HTMLDivElement>(null);

  // Use a ref to store the latest attachments and avoid race conditions with state updates
  const latestAttachmentsRef = useRef<FileAttachment[]>([]);

  // Ref for the end of messages container for scrolling
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [pendingAttachments, setPendingAttachments] = useState<FileAttachment[]>([]);

  // Add a ref for the last user message
  const lastUserMessageRef = useRef<HTMLDivElement>(null);

  // Add a new style for the last user message element
  const lastUserMessageStyle = {
    scrollMarginTop: '16px' // Same as the padding between cards
  };

  // Function to scroll to the last user message
  const scrollToLastUserMessage = useCallback(() => {
    if (lastUserMessageRef.current) {
      // Use scrollIntoView with the element that has scroll margin applied
      lastUserMessageRef.current.scrollIntoView({ 
        behavior: 'smooth',
        block: 'start' // Align to the top of the viewport
      });
      console.log('Scrolled to last user message with padding');
    } else {
      // Fallback to bottom if no user message is found
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      console.log('No user message found, scrolled to bottom');
    }
  }, []);

  // Function to poll for task status updates after taking an action
  const pollTaskStatus = useCallback(
    async (taskId: string, attempts = 5, delay = 1000) => {
      // Initial delay before first poll
      await new Promise(resolve => setTimeout(resolve, delay));

      for (let i = 0; i < attempts; i++) {
        try {
          console.log(`Polling task status attempt ${i + 1} of ${attempts}`);
          const response = await agentService.getTask(taskId);
          if ('error' in response) {
            throw new Error(response.error);
          }
          const fetchedTask = response as Task;
          setTask(fetchedTask);
          
          // If task is in a terminal state, stop polling
          if (fetchedTask.status === TaskStatus.COMPLETED || fetchedTask.status === TaskStatus.ERROR) {
            break;
          }
        } catch (error) {
          console.error('Error polling task status:', error);
        }
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    },
    [setTask]
  );

  // Load messages when taskId changes
  useEffect(() => {
    const loadMessages = async () => {
      if (!taskId) {
        console.log('No taskId provided, skipping message loading');
        return;
      }
      
      console.log('Loading messages for taskId:', taskId);
      setIsLoading(true);
      try {
        // Use agentService instead of direct fetch
        const data = await agentService.getTaskMessages(taskId);
        
        // Convert array to object with message IDs as keys
        const messageEntriesObj: Record<string, TaskMessageEntry> = {};
        data.forEach((entry: TaskMessageEntry) => {
          if (entry.id) {
            // Messages are already in the correct format from the backend
            messageEntriesObj[entry.id] = entry;
          } else {
            console.warn('Message entry without ID found:', entry);
          }
        });
        setMessageEntries(messageEntriesObj);
        previousMessagesLength.current = Object.keys(messageEntriesObj).length;
        
        // After messages are loaded, scroll to last user message
        setTimeout(() => {
          scrollToLastUserMessage();
        }, 100);
      } catch (error) {
        console.error('Error loading messages:', error);
        toast({
          title: 'Error',
          description: 'Failed to load messages. Please try again.',
          variant: 'destructive',
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadMessages();
  }, [taskId, toast, scrollToLastUserMessage]);

  // Load task data
  useEffect(() => {
    if (!taskId) return;

    const loadTaskData = async () => {
      try {
        console.log('Loading task data for:', taskId);
        const response = await agentService.getTask(taskId);
        if ('error' in response) {
          console.error('Error loading task:', response.error);
          return;
        }
        const taskData = response as Task;
        setTask(taskData);
        setSelectedTask(taskData);
      } catch (error) {
        console.error('Error loading task data:', error);
      }
    };

    loadTaskData();
  }, [taskId, setSelectedTask]);

  // Function to scroll to the bottom of messages
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  // Function to scroll so that the newest message is at the top of the viewport
  const scrollToNewMessage = useCallback(() => {
    if (newestMessageRef.current && messagesContainerRef.current) {
      const containerRect = messagesContainerRef.current.getBoundingClientRect();
      const offsetTop = newestMessageRef.current.offsetTop;
      
      // Scroll to position the newest message at the top of the viewport with some padding
      messagesContainerRef.current.scrollTo({
        top: offsetTop - 20, // 20px padding from the top
        behavior: 'smooth'
      });
      
      console.log('Scrolling to new message at position:', offsetTop);
    } else {
      console.warn('Cannot scroll to new message - refs not available');
    }
  }, []);

  // Updated scroll behavior when new messages are added
  useEffect(() => {
    const currentMessagesCount = Object.keys(messageEntries).length;
    
    // Only scroll if messages length actually increased and autoscroll is enabled
    if (currentMessagesCount > previousMessagesLength.current && autoScrollEnabled) {
      // Find if the newest message is from the agent
      const messagesArray = Object.values(messageEntries);
      if (messagesArray.length > 0) {
        const newestMessage = messagesArray[messagesArray.length - 1];
        if (newestMessage.message.author === MessageAuthor.AGENT) {
          // Use the new scrolling behavior for agent messages
          setTimeout(scrollToNewMessage, 100); // Small delay to ensure the DOM is updated
        }
      }
    }
    
    // Always update the previous length even if we don't scroll
    previousMessagesLength.current = currentMessagesCount;
  }, [Object.keys(messageEntries).length, scrollToNewMessage, autoScrollEnabled]);

  // Handle task approval
  const handleApprove = async () => {
    if (!taskId) return;
    setIsApproving(true);
    try {
      await agentService.sendSignal(taskId, {
        type: TaskSignalType.APPROVE
      });
      pollTaskStatus(taskId);
    } catch (error) {
      console.error('Error approving task:', error);
      toast({
        title: 'Error',
        description: 'Failed to approve task',
        variant: 'destructive',
      });
    } finally {
      setIsApproving(false);
    }
  };

  // Handle task cancellation
  const handleCancel = async () => {
    if (!taskId) return;
    setIsCancelling(true);
    try {
      await agentService.sendSignal(taskId, {
        type: TaskSignalType.CANCEL
      });
      pollTaskStatus(taskId);
    } catch (error) {
      console.error('Error cancelling task:', error);
      toast({
        title: 'Error',
        description: 'Failed to cancel task',
        variant: 'destructive',
      });
    } finally {
      setIsCancelling(false);
    }
  };

  // Handle file attachments
  const handleAttachmentsChange = (
    attachments: FileAttachment[] | ((curr: FileAttachment[]) => FileAttachment[])
  ) => {
    if (typeof attachments === 'function') {
      const newAttachments = attachments(latestAttachmentsRef.current);
      setPendingAttachments(newAttachments);
      latestAttachmentsRef.current = newAttachments;
    } else {
      setPendingAttachments(attachments);
      latestAttachmentsRef.current = attachments;
    }
  };

  // Update handleSubmit to scroll after sending user message
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Get attachments from both sources
    const eventAttachments = ((e as any).attachments || []) as FileAttachment[];
    const refAttachments = latestAttachmentsRef.current;

    // Use eventAttachments if available, otherwise use refAttachments
    const allFileAttachments = eventAttachments.length > 0 ? eventAttachments : refAttachments;

    // Convert to MessageFileAttachment
    const messageAttachments = allFileAttachments
      .filter(att => att.file_id && att.status === 'success')
      .map(att => ({
        file_id: att.file_id,
        name: att.name,
        size: att.size,
        type: att.type,
        status: 'success' as const
      }));

    if (!userInput.trim() && messageAttachments.length === 0) return;

    // Add debug logs
    console.log('Submitting message with messageAttachments:', messageAttachments.length);
    console.log('Event attachments:', eventAttachments.length);
    console.log('Ref attachments:', refAttachments.length);
    console.log('State attachments (pendingAttachments):', pendingAttachments.length);

    // Log details of each attachment to help debugging
    if (messageAttachments.length > 0) {
      console.log('Attachment details:');
      messageAttachments.forEach((att, i) => {
        console.log(`  [${i}] ${att.name} (${att.file_id})`);
      });
    }

    setUserInput('');
    setIsThinking(true);

    try {
      if (task?.id) {
        // Create a TaskMessage object
        const message: TextMessage = {
          type: MessageType.TEXT,
          author: MessageAuthor.USER,
          style: MessageStyle.STATIC,
          format: TextFormat.PLAIN,
          content: userInput,
          attachments: messageAttachments.length > 0 ? messageAttachments : undefined,
        };

        // Log the final message being sent
        console.log('Sending message to workflow:', JSON.stringify(message, null, 2));

        // Check if we have an active SSE connection before sending the message
        console.log('Checking SSE connection state before sending message...');
        const hasActiveStream = agentService.hasActiveStream(task.id);
        console.log(`Active SSE stream for task ${task.id}: ${hasActiveStream}`);
        
        // Send the signal to the workflow with the message
        const response = await agentService.signalTask(task.id, {
          type: TaskSignalType.INSTRUCT,
          message
        });
        
        console.log('Signal response:', response);
        
        // Check if we still have an active SSE connection after sending the message
        console.log('Checking SSE connection state after sending message...');
        const hasActiveStreamAfter = agentService.hasActiveStream(task.id);
        console.log(`Active SSE stream for task ${task.id} after sending: ${hasActiveStreamAfter}`);
        
        // If we don't have an active stream, reconnect
        if (!hasActiveStreamAfter) {
          console.log('No active SSE stream after sending message, reconnecting...');
          connectToStream();
        }

        // Clear pending attachments after submission
        setPendingAttachments([]);
        latestAttachmentsRef.current = [];

        // Scroll to bottom after sending user message
        if (autoScrollEnabled) {
          setTimeout(scrollToBottom, 100);
        }
      }
    } catch (err) {
      console.error('Error in form submission:', err);
    } finally {
      setIsThinking(false);
    }
  };

  // Render messages
  const renderMessages = () => {
    const messagesArray = Object.values(messageEntries);
    
    // Sort messages by timestamp if available
    messagesArray.sort((a, b) => {
      const aTime = a.timestamp ? new Date(a.timestamp).getTime() : 0;
      const bTime = b.timestamp ? new Date(b.timestamp).getTime() : 0;
      return aTime - bTime;
    });
    
    // Find the last user message index
    let lastUserMessageIndex = -1;
    for (let i = messagesArray.length - 1; i >= 0; i--) {
      const msg = messagesArray[i]?.message as any;
      if (msg && msg.author === MessageAuthor.USER) {
        lastUserMessageIndex = i;
        break;
      }
    }
    
    return messagesArray.map((entry, index) => {
      const message = entry?.message as any;
      if (!message) {
        return null;
      }
      
      // Add a ref to the most recent message from the agent
      const isNewestAgentMessage = index === messagesArray.length - 1 && 
                                   message?.author === MessageAuthor.AGENT;
                                   
      // Add ref to the last user message
      const isLastUserMessage = index === lastUserMessageIndex;

      if (message.type === MessageType.TOOL_REQUEST) {
        const toolRequest = message as ToolRequestMessage;
        return (
          <div 
            key={entry.id}
            ref={isNewestAgentMessage ? newestMessageRef : 
                isLastUserMessage ? lastUserMessageRef : undefined}
            style={isLastUserMessage ? lastUserMessageStyle : undefined}
          >
            <ToolMessageCard
              type="request"
              name={toolRequest.name}
              content={toolRequest.arguments}
              style={message.style === MessageStyle.ACTIVE ? 'active' : 'static'}
            />
          </div>
        );
      } else if (message.type === MessageType.TOOL_RESPONSE) {
        const toolResponse = message as ToolResponseMessage;
        return (
          <div 
            key={entry.id}
            ref={isNewestAgentMessage ? newestMessageRef : 
                isLastUserMessage ? lastUserMessageRef : undefined}
            style={isLastUserMessage ? lastUserMessageStyle : undefined}
          >
            <ToolMessageCard
              type="response"
              name={toolResponse.name}
              content={toolResponse.content}
              style={message.style === MessageStyle.ACTIVE ? 'active' : 'static'}
            />
          </div>
        );
      }

      let content = '';
      let attachments: FileAttachment[] | undefined;
      
      if (message.type === MessageType.TEXT) {
        const textMessage = message as TextMessage;
        content = textMessage.content;
        attachments = textMessage.attachments;
      } else if (message.type === MessageType.JSON) {
        const jsonMessage = message as JSONMessage;
        content = JSON.stringify(jsonMessage.data, null, 2);
      }

      return (
        <div 
          key={entry.id}
          ref={isNewestAgentMessage ? newestMessageRef : 
              isLastUserMessage ? lastUserMessageRef : undefined}
          style={isLastUserMessage ? lastUserMessageStyle : undefined}
        >
          <MessageCard
            content={content}
            role={message?.author === MessageAuthor.USER ? 'user' : 'agent'}
            style={message?.style === MessageStyle.ACTIVE ? 'active' : 'static'}
            attachments={attachments}
          />
        </div>
      );
    });
  };

  const copyTaskId = () => {
    navigator.clipboard.writeText(taskId).then(() => {
      setIsCopied(true);
      toast({
        title: 'Copied!',
        description: `Task ID ${taskId} copied to clipboard`,
      });
      setTimeout(() => setIsCopied(false), 2000);
    });
  };

  const taskInTerminalState =
    task?.status === TaskStatus.ERROR || task?.status === TaskStatus.COMPLETED || task?.status === TaskStatus.CANCELLED;

  // Connect to SSE stream for real-time updates
  const connectToStream = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current();
    }

    const handleStreamEvent: StreamEventCallback = (data: TaskEvent) => {
      console.log('Received stream event:', data);
      
      if (data.type === EventType.STREAMING_CHUNK) {
        console.log('Processing streaming_chunk event:', data);
        
        const messageId = data.parent_task_message_entry?.id;
        if (messageId) {
          setMessageEntries(prevEntries => {
            const updatedEntries = { ...prevEntries };
            if (!updatedEntries.hasOwnProperty(messageId)) {
              const parentEntry = data.parent_task_message_entry;
              updatedEntries[messageId] = {
                ...parentEntry,
                message: {
                  type: MessageType.TEXT,
                  format: TextFormat.PLAIN,
                  content: (parentEntry.message as TextMessage).content || '',
                  author: parentEntry.message.author,
                  style: parentEntry.message.style,
                  attachments: (parentEntry.message as TextMessage).attachments,
                } as TextMessage,
              };
            }

            const currentEntry = updatedEntries[messageId];
            const currentContent = (currentEntry.message as TextMessage).content || '';
            updatedEntries[messageId] = {
              ...currentEntry,
              message: {
                ...currentEntry.message,
                content: currentContent + (data.content_chunk || ''),
              } as TextMessage,
            };
            return updatedEntries;
          });
        } else {
          console.warn('Received streaming_chunk without messageId:', data);
        }
      } else if (data.type === EventType.TOOL_REQUEST) {
        const toolRequest = data as ToolRequestEvent;
        const messageId = toolRequest.parent_task_message_entry?.id;
        if (messageId) {
          setMessageEntries(prevEntries => {
            const updatedEntries = { ...prevEntries };
            if (!updatedEntries.hasOwnProperty(messageId)) {
              const parentEntry = toolRequest.parent_task_message_entry;
              updatedEntries[messageId] = {
                ...parentEntry,
                message: {
                  type: MessageType.TOOL_REQUEST,
                  name: toolRequest.name,
                  arguments: toolRequest.arguments,
                  author: parentEntry.message.author,
                  style: parentEntry.message.style,
                } as ToolRequestMessage,
              };
            }
            return updatedEntries;
          });
        }
      } else if (data.type === EventType.TOOL_RESPONSE) {
        const toolResponse = data as ToolResponseEvent;
        const messageId = toolResponse.parent_task_message_entry?.id;
        if (messageId) {
          setMessageEntries(prevEntries => {
            const updatedEntries = { ...prevEntries };
            if (!updatedEntries.hasOwnProperty(messageId)) {
              const parentEntry = toolResponse.parent_task_message_entry;
              updatedEntries[messageId] = {
                ...parentEntry,
                message: {
                  type: MessageType.TOOL_RESPONSE,
                  name: toolResponse.name,
                  content: toolResponse.content,
                  author: parentEntry.message.author,
                  style: parentEntry.message.style,
                } as ToolResponseMessage,
              };
            }
            return updatedEntries;
          });
        }
      } else if (data.type === EventType.FULL_CONTENT_UPDATE) {
        const { parent_task_message_entry, content } = data;
        if (parent_task_message_entry?.id) {
          const messageId = parent_task_message_entry.id;
          setMessageEntries(prevEntries => {
            const updatedEntries = { ...prevEntries };
            updatedEntries[messageId] = {
              ...parent_task_message_entry,
              message: {
                type: MessageType.TEXT,
                format: TextFormat.PLAIN,
                content: content || '',
                author: parent_task_message_entry.message.author,
                style: parent_task_message_entry.message.style,
                attachments: (parent_task_message_entry.message as TextMessage).attachments,
              } as TextMessage,
            };
            return updatedEntries;
          });
        } else {
          console.warn('Received full_content_update without messageId:', data);
        }
      } else if (data.type === EventType.ERROR) {
        console.error('Received error event:', data.error_message);
      } else if (data.type === EventType.DONE) {
        console.log('Received done event');
      } else {
        // This should never happen due to the discriminated union, but TypeScript doesn't know that
        console.log('Unhandled event type:', (data as any).type);
      }
    };

    const handleError: ErrorCallback = (error) => {
      console.error('EventSource error:', error);
      // Try to reconnect after a delay
      setTimeout(() => {
        console.log('Attempting to reconnect to SSE stream...');
        connectToStream();
      }, 3000);
    };

    const handleConnected: ConnectionCallback = () => {
      console.log('SSE connection established');
    };

    const handleCompleted: CompletionCallback = (taskId) => {
      console.log(`Stream completed for task: ${taskId}`);
    };

    eventSourceRef.current = agentService.connectToStream(
      taskId,
      handleStreamEvent,
      handleError,
      handleConnected,
      handleCompleted
    );
  }, [taskId]);

  useEffect(() => {
    // Initial load of messages
    connectToStream();

    // Cleanup function
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current();
      }
    };
  }, [taskId, connectToStream]);

  // Monitor SSE connection state and reconnect if needed
  useEffect(() => {
    if (!taskId) return;
    
    // Set up an interval to check the connection state
    const intervalId = setInterval(() => {
      const hasActiveStream = agentService.hasActiveStream(taskId);
      // console.log(`SSE connection check for task ${taskId}: ${hasActiveStream ? 'active' : 'inactive'}`);
      
      if (!hasActiveStream) {
        console.log('No active SSE stream detected, reconnecting...');
        connectToStream();
      }
    }, 5000); // Check every 5 seconds
    
    // Clean up the interval when the component unmounts
    return () => {
      clearInterval(intervalId);
    };
  }, [taskId, connectToStream]);

  // Add effect to scroll to last user message when component mounts
  useEffect(() => {
    // Scroll to last user message when the component mounts or refreshes
    const initialScrollTimeout = setTimeout(() => {
      scrollToLastUserMessage();
    }, 500); // Small delay to ensure DOM is fully rendered
    
    return () => clearTimeout(initialScrollTimeout);
  }, [scrollToLastUserMessage]);

  return (
    <ResizablePanelGroup direction="horizontal" className="h-full bg-white">
      <ResizablePanel defaultSize={100} className="h-full">
        <div className="h-full flex flex-col">
          {task && (
            <>
              <div className="flex-grow overflow-y-auto" ref={messagesContainerRef}>
                <div className="max-w-5xl mx-auto p-4 pt-8 space-y-4">
                  {isLoading ? (
                    <div className="flex justify-center items-center h-32">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900" />
                    </div>
                  ) : (
                    <>
                      {pendingAttachments.length > 0 && (
                        <div className="flex justify-between mb-2">
                          <div className="text-sm text-gray-500">
                            {pendingAttachments.length} file{pendingAttachments.length !== 1 ? 's' : ''}{' '}
                            pending submission
                          </div>
                        </div>
                      )}
                      <AnimatePresence>{renderMessages()}</AnimatePresence>
                      {/* Add extra space at the bottom to allow room for streaming text */}
                      {/* <div className="h-[60vh]" /> */}
                      <div ref={messagesEndRef} />
                    </>
                  )}
                </div>
                <div className="sticky bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-white via-white/60 to-transparent pointer-events-none" />

                {/* Scroll to bottom button - only visible when autoscroll is disabled */}
                {!autoScrollEnabled && Object.keys(messageEntries).length > 0 && (
                  <div className="fixed bottom-28 right-8 z-10">
                    <Button
                      size="sm"
                      className="rounded-full shadow-md p-3 h-10 w-10"
                      onClick={scrollToBottom}
                    >
                      <ArrowDown className="h-4 w-4" />
                      <span className="sr-only">Scroll to bottom</span>
                    </Button>
                  </div>
                )}
              </div>
              <div className="relative">
                {/* {pendingAttachments.length > 0 && (
                  <div className="max-w-5xl w-full mx-auto px-4 mb-2">
                    <div className="p-2 bg-yellow-50 border border-yellow-200 rounded-md">
                      <div className="flex justify-between items-center">
                        <p className="text-sm text-yellow-700 font-medium">
                          {pendingAttachments.length} file
                          {pendingAttachments.length !== 1 ? 's' : ''} will be sent with your next
                          message:
                        </p>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="text-yellow-700 hover:text-yellow-900 text-xs"
                          onClick={() => setPendingAttachments([])}
                        >
                          Clear All
                        </Button>
                      </div>
                      <ul className="text-xs text-yellow-600 mt-1">
                        {pendingAttachments.map(att => (
                          <li key={att.file_id}>{att.name}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )} */}
                <div className="animate-fade-in-up">
                  <TaskBadges
                    isCopied={isCopied}
                    task={task}
                    copyTaskId={copyTaskId}
                    autoScrollEnabled={autoScrollEnabled}
                    setAutoScrollEnabled={setAutoScrollEnabled}
                  />
                  <TaskControls
                    userInput={userInput}
                    setUserInput={setUserInput}
                    handleSubmit={handleSubmit}
                    isThinking={isThinking}
                    taskInTerminalState={
                      task?.status === TaskStatus.COMPLETED || task?.status === TaskStatus.ERROR || false
                    }
                    handleApproveTask={handleApprove}
                    isApproving={isThinking}
                    handleCancelTask={handleCancel}
                    isCancelling={isCancelling}
                    onAttachmentsChange={handleAttachmentsChange}
                  />
                </div>
              </div>
            </>
          )}
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

export default Page;
