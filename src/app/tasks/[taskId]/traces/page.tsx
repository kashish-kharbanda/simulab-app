'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { GanttChart, type Span } from '@/components/tracing/GanttChart';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, X, ArrowLeft, RefreshCw, Clock, Download } from 'lucide-react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { useParams } from 'next/navigation';
import Link from 'next/link';
import { useToast } from '@/hooks/use-toast';
import { agentService } from '@/services/agentService';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Task, TaskResponse } from '@/types/tasks';

// Empty data for initial state
const emptyData: Span[] = [];

// Add this component at the top level, outside the main component
// This ensures the animation stays consistent and doesn't cause re-renders
function PulsingIndicator() {
  return (
    <span
      className="ml-1.5 h-2 w-2 rounded-full bg-green-500"
      style={{
        animation: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }}
    />
  );
}

export default function TraceViewer() {
  const params = useParams();
  const taskId = params.taskId as string;
  const { toast } = useToast();
  const [task, setTask] = useState<Task | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [spans, setSpans] = useState<Span[]>(emptyData);
  const [selectedSpan, setSelectedSpan] = useState<Span | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [traceId, setTraceId] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);
  const [isManualLoading, setIsManualLoading] = useState(false);

  // Polling control refs
  const pollingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isPollingLockedRef = useRef<boolean>(false);
  const lastPollTimeRef = useRef<number>(0);
  const previousSpanCountRef = useRef<number>(0);
  const selectedSpanIdRef = useRef<string | null>(null);

  // Polling configuration
  const STRICT_POLL_INTERVAL = 2000; // Exactly 2 seconds
  const MIN_POLL_INTERVAL = 1900; // Absolute minimum time between refreshes

  useEffect(() => {
    async function loadTask() {
      try {
        const response = await agentService.getTask(taskId);
        if ('error' in response) {
          throw new Error(response.error);
        }
        setTask(response as Task);
        setError(null);
      } catch (err) {
        console.error('Error loading task:', err);
        setError('Failed to load task');
        toast({
          title: 'Error',
          description: 'Failed to load task',
          variant: 'destructive',
        });
      } finally {
        setIsLoading(false);
      }
    }

    loadTask();
  }, [taskId, toast]);

  // Function to handle toggling polling state - the only place that should toggle polling
  const togglePolling = useCallback(() => {
    // If turning off polling, immediately clear any pending timeouts
    if (isPolling) {
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
      }
    }
    setIsPolling(prev => !prev);
  }, [isPolling]);

  // Function to fetch task trace ID
  const fetchTaskTraceId = useCallback(async () => {
    try {
      const response = await agentService.getTask(taskId);
      if ('error' in response) {
        throw new Error(response.error);
      }
      
      // Use taskId as the trace ID
      setTraceId(taskId);
      return taskId;
    } catch (error) {
      console.error('Error fetching task trace ID:', error);
      // Return taskId as fallback
      setTraceId(taskId);
      return taskId;
    }
  }, [taskId]);

  // Function to fetch trace spans
  const fetchTraceSpans = useCallback(async (tid: string) => {
    try {
      // Use the agentService to fetch spans
      return await agentService.getTaskSpans(taskId, tid);
    } catch (err) {
      throw err;
    }
  }, [taskId]);

  // Update selectedSpanIdRef whenever selectedSpan changes
  useEffect(() => {
    selectedSpanIdRef.current = selectedSpan?.id || null;
  }, [selectedSpan]);

  // Main function to load trace data
  const loadTraceData = useCallback(
    async (isManualRefresh = false) => {
      // GUARD: Check if polling is locked (another request is in progress)
      if (isPollingLockedRef.current) {
        console.log('[STRICT] Skipping poll - already polling');
        return;
      }

      // GUARD: Enforce strict time interval between polls
      const now = Date.now();
      const timeSinceLastPoll = now - lastPollTimeRef.current;

      if (!isManualRefresh && timeSinceLastPoll < MIN_POLL_INTERVAL) {
        console.log(
          `[STRICT] Poll too soon (${timeSinceLastPoll}ms since last poll). Enforcing rate limit.`,
        );
        return;
      }

      // LOCK: Set the polling lock and update time
      isPollingLockedRef.current = true;
      lastPollTimeRef.current = now;

      // Set loading states appropriately
      if (isManualRefresh) {
        setIsManualLoading(true);
      }
      if (!isPolling || isManualRefresh) {
        setIsLoading(true);
      }
      setError(null);

      try {
        // First get the trace ID if we don't have it yet
        const tid = traceId || (await fetchTaskTraceId());

        // Then fetch the spans for that trace ID
        const spans = await fetchTraceSpans(tid);

        // Keep track of previous span count to detect changes
        const currentSpanCount = spans.length;
        const previousSpanCount = previousSpanCountRef.current;

        // Get current selected span ID from ref to avoid dependency issues
        const currentSelectedSpanId = selectedSpanIdRef.current;

        setSpans(spans);
        // Update the previous span count ref
        previousSpanCountRef.current = currentSpanCount;

        // If there was a selected span, try to find and reselect it in the new data
        if (currentSelectedSpanId) {
          const updatedSelectedSpan = spans.find(span => span.id === currentSelectedSpanId);
          if (updatedSelectedSpan) {
            // During auto-refresh polls, don't update the selectedSpan state to prevent re-renders
            if (isManualRefresh || !isPolling) {
              setSelectedSpan(updatedSelectedSpan);
            }
          }
        }

        // When auto-refreshing, only show a toast if the span count has changed
        if (isPolling && !isManualRefresh) {
          if (currentSpanCount !== previousSpanCount) {
            toast({
              title: 'New Spans Detected',
              description: `Updated from ${previousSpanCount} to ${currentSpanCount} spans`,
              duration: 2000,
            });
          }
        } else {
          // Always show a toast for manual refresh
          toast({
            title: 'Trace Data Loaded',
            description: `Successfully loaded ${currentSpanCount} spans`,
          });
        }
      } catch (err) {
        console.error('Error loading trace data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load trace data');
        toast({
          variant: 'destructive',
          title: 'Error Loading Traces',
          description: err instanceof Error ? err.message : 'Failed to load trace data',
        });

        // Set empty data on error
        setSpans(emptyData);
        previousSpanCountRef.current = 0;
      } finally {
        // Only update loading states if this is not an auto-refresh or we're doing a manual refresh
        if (isManualRefresh) {
          setIsManualLoading(false);
        }
        if (!isPolling || isManualRefresh) {
          setIsLoading(false);
        }

        // Release the polling lock
        isPollingLockedRef.current = false;
      }
    },
    // Remove selectedSpan from dependencies - we use the ref instead
    [traceId, toast, fetchTaskTraceId, fetchTraceSpans, isPolling],
  );

  // Load trace data on initial render
  useEffect(() => {
    loadTraceData(true);
  }, [taskId, loadTraceData]);

  // Single, controlled polling effect
  useEffect(() => {
    let isActive = true; // Closure variable to prevent stale callbacks

    // Setup function for polling
    const setupPolling = () => {
      // Clear any existing timeout first
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
      }

      // If not polling, don't set up a new timer
      if (!isPolling || !isActive) return;

      // Single poll function that also schedules the next poll
      const poll = () => {
        // Safety checks - only proceed if polling is still enabled and component is mounted
        if (!isPolling || !isActive) {
          console.log('[STRICT] Polling stopped or component unmounted');
          return;
        }

        // Additional safety check for rapid consecutive calls
        const now = Date.now();
        const timeSinceLastPoll = now - lastPollTimeRef.current;
        if (timeSinceLastPoll < MIN_POLL_INTERVAL) {
          console.log(`[STRICT-POLL] Too soon to poll (${timeSinceLastPoll}ms). Rescheduling.`);
          // Reschedule after the minimum interval has passed
          pollingTimeoutRef.current = setTimeout(poll, MIN_POLL_INTERVAL - timeSinceLastPoll + 100);
          return;
        }

        console.log('[STRICT] Executing controlled poll');

        // Execute the data load - only if we're not already loading
        if (!isPollingLockedRef.current) {
          loadTraceData(false).finally(() => {
            // After loading completes (success or failure), schedule next poll
            // but only if we're still in polling mode and component is mounted
            if (isPolling && isActive) {
              console.log('[STRICT] Scheduling next poll in exactly 2 seconds');
              pollingTimeoutRef.current = setTimeout(poll, STRICT_POLL_INTERVAL);
            }
          });
        } else {
          // If already loading, reschedule after a delay
          console.log('[STRICT] Previous poll still running, rescheduling');
          pollingTimeoutRef.current = setTimeout(poll, 1000);
        }
      };

      // Start the first poll with a small delay to let things settle
      console.log('[STRICT] Starting initial poll');
      pollingTimeoutRef.current = setTimeout(poll, 500);
    };

    // Handle polling state changes
    if (isPolling) {
      // Notify user that polling has started
      toast({
        title: 'Auto-refresh enabled',
        description: 'Trace data will update every 2 seconds',
        duration: 3000,
      });

      // Start polling
      setupPolling();
    } else {
      // Notify user that polling has stopped
      if (pollingTimeoutRef.current) {
        toast({
          title: 'Auto-refresh disabled',
          description: 'Automatic updates have been stopped',
          duration: 3000,
        });

        // Clean up any existing timeout
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
      }
    }

    // Cleanup on unmount or when dependencies change
    return () => {
      isActive = false; // Mark as inactive for closure safety
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
      }
    };
  }, [isPolling, loadTraceData, toast]);

  // Also clean up on component unmount
  useEffect(() => {
    return () => {
      if (pollingTimeoutRef.current) {
        clearTimeout(pollingTimeoutRef.current);
        pollingTimeoutRef.current = null;
      }
    };
  }, []);

  // Handle span selection separately from polling
  const handleSpanSelect = useCallback((span: Span) => {
    // Only change the selected span, do NOT toggle polling
    setSelectedSpan(span);
    // Update the ref immediately to avoid timing issues
    selectedSpanIdRef.current = span.id;
  }, []);

  // Format time difference as ms, μs, or ns
  function formatDuration(ms: number) {
    if (ms >= 1000) return `${(ms / 1000).toFixed(2)}s`;
    if (ms >= 1) return `${ms.toFixed(2)}ms`;
    if (ms >= 0.001) return `${(ms * 1000).toFixed(2)}μs`;
    return `${(ms * 1000000).toFixed(2)}ns`;
  }

  // Format absolute time
  function formatTime(timestamp: Date) {
    return `${timestamp.toLocaleTimeString()}.${timestamp
      .getMilliseconds()
      .toString()
      .padStart(3, '0')}`;
  }

  // Function to handle downloading spans as JSON
  const handleDownloadSpans = () => {
    // Create a JSON blob from the spans data
    const dataStr = JSON.stringify(spans, null, 2);
    const blob = new Blob([dataStr], { type: 'application/json' });

    // Create a download URL for the blob
    const url = URL.createObjectURL(blob);

    // Create a temporary anchor element to trigger the download
    const a = document.createElement('a');
    a.href = url;
    a.download = `trace-${traceId?.substring(0, 8) || 'data'}-spans.json`;
    document.body.appendChild(a);
    a.click();

    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    toast({
      title: 'Download started',
      description: `Downloading ${spans.length} spans as JSON`,
      duration: 2000,
    });
  };

  return (
    <main className="container mx-auto py-6 px-4 flex flex-col h-screen">
      <div className="flex justify-between items-center mb-6">
        <div className="flex items-center">
          <Link href={`/tasks/${taskId}`}>
            <Button variant="outline" size="sm" className="mr-2 flex items-center gap-1">
              <ArrowLeft className="h-4 w-4" />
              Back to Chat
            </Button>
          </Link>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Switch
              id="auto-refresh"
              checked={isPolling}
              onCheckedChange={togglePolling}
              disabled={isManualLoading}
              aria-label="Toggle auto-refresh"
            />
            <Label htmlFor="auto-refresh" className="flex items-center cursor-pointer">
              {isPolling ? (
                <span className="flex items-center">
                  Auto-refreshing
                  <PulsingIndicator />
                </span>
              ) : (
                'Auto-reload'
              )}
            </Label>
          </div>
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {traceId && (
        <Alert className="mb-4">
          <div className="flex justify-between items-center w-full">
            <div>
              <div className="flex items-center gap-1">
                <p className="text-sm font-medium">
                  Trace ID:
                  <span className="inline-block" title={traceId}>
                    {' '}
                    {traceId ? `${traceId.substring(0, 8)}...` : ''}
                  </span>
                </p>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(traceId || '');
                    toast({
                      title: 'Copied to clipboard',
                      description: 'Trace ID has been copied to clipboard',
                      duration: 2000,
                    });
                  }}
                  className="text-gray-500 hover:text-gray-700 ml-1 focus:outline-none p-1 rounded-md hover:bg-gray-100"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="feather feather-copy"
                  >
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                  </svg>
                  <span className="sr-only">Copy trace ID</span>
                </button>
              </div>
              <p className="text-xs text-muted-foreground">
                Showing spans associated with this trace
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span>{spans.length} spans</span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownloadSpans}
                disabled={spans.length === 0}
                className="flex items-center gap-1"
              >
                <Download className="h-3.5 w-3.5" />
                <span className="text-xs">Download</span>
              </Button>
            </div>
          </div>
        </Alert>
      )}

      {/* Main Content Area - Gantt Chart and Detail Panel side by side */}
      <div className="flex flex-1 gap-4 mb-4 min-h-0">
        {/* Gantt Chart - Left Section */}
        <Card className="flex-1 min-w-0 flex flex-col">
          <CardHeader className="py-3">
            <CardTitle className="text-lg">Gantt Chart</CardTitle>
            <CardDescription>
              Hierarchical view of spans - click arrows to expand/collapse
            </CardDescription>
          </CardHeader>
          <CardContent className="p-2 flex-1 min-h-0 overflow-hidden">
            {spans.length > 0 ? (
              <GanttChart
                spans={spans}
                onSpanSelect={handleSpanSelect} // This should only select spans, never affect polling
                selectedSpanId={selectedSpan?.id}
                autoScrollToBottom={isPolling}
              />
            ) : (
              <div className="flex items-center justify-center h-full border rounded-md">
                <p className="text-muted-foreground">No trace data available</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Span Details - Right Section (always visible with empty state when no span selected) */}
        <Card className="w-[450px] flex flex-col min-w-0">
          {selectedSpan ? (
            <>
              <CardHeader className="py-3 flex-shrink-0">
                <div className="flex justify-between items-center">
                  <CardTitle className="text-lg truncate" title={selectedSpan.name}>
                    {selectedSpan.name}
                  </CardTitle>
                  <Button variant="ghost" size="icon" onClick={() => setSelectedSpan(null)}>
                    <X className="h-4 w-4" />
                    <span className="sr-only">Close</span>
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-2 flex-1 min-h-0 overflow-hidden">
                <div className="flex flex-col h-full">
                  <div className="grid grid-cols-2 gap-4 mb-4 flex-shrink-0">
                    <div>
                      <p className="text-sm font-medium mb-1">Trace ID</p>
                      <div className="flex items-center gap-1">
                        <p className="text-sm" title={selectedSpan.trace_id}>
                          {selectedSpan.trace_id
                            ? `${selectedSpan.trace_id.substring(0, 8)}...`
                            : ''}
                        </p>
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(selectedSpan.trace_id);
                            toast({
                              title: 'Copied to clipboard',
                              description: 'Trace ID has been copied to clipboard',
                              duration: 2000,
                            });
                          }}
                          className="text-gray-500 hover:text-gray-700 ml-1 focus:outline-none p-1 rounded-md hover:bg-gray-100"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="14"
                            height="14"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="feather feather-copy"
                          >
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                          </svg>
                          <span className="sr-only">Copy trace ID</span>
                        </button>
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-1">Duration</p>
                      <p>
                        {formatDuration(
                          (selectedSpan.end_time
                            ? new Date(selectedSpan.end_time)
                            : new Date()
                          ).getTime() - new Date(selectedSpan.start_time).getTime(),
                        )}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-1">Start Time</p>
                      <p>{formatTime(new Date(selectedSpan.start_time))}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium mb-1">End Time</p>
                      <p>
                        {selectedSpan.end_time
                          ? formatTime(new Date(selectedSpan.end_time))
                          : 'Ongoing'}
                      </p>
                    </div>
                    {selectedSpan.parent_id && (
                      <>
                        <div>
                          <p className="text-sm font-medium mb-1">Parent ID</p>
                          {/* Check if parent span exists in the data */}
                          {spans.some(span => span.id === selectedSpan.parent_id) ? (
                            <div className="flex items-center gap-1">
                              <button
                                onClick={() => {
                                  const parentSpan = spans.find(
                                    span => span.id === selectedSpan.parent_id,
                                  );
                                  if (parentSpan) {
                                    handleSpanSelect(parentSpan);
                                  }
                                }}
                                className="bg-blue-100 text-blue-700 px-2 py-0.5 rounded hover:bg-blue-200 transition-colors focus:outline-none text-left flex items-center text-sm"
                              >
                                <span
                                  className="truncate"
                                  title={
                                    spans.find(span => span.id === selectedSpan.parent_id)?.name ||
                                    selectedSpan.parent_id
                                  }
                                >
                                  {spans.find(span => span.id === selectedSpan.parent_id)?.name ||
                                    (selectedSpan.parent_id
                                      ? `${selectedSpan.parent_id.substring(0, 8)}...`
                                      : '')}
                                </span>
                              </button>
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(selectedSpan.parent_id || '');
                                  toast({
                                    title: 'Copied to clipboard',
                                    description: 'Parent ID has been copied to clipboard',
                                    duration: 2000,
                                  });
                                }}
                                className="text-gray-500 hover:text-gray-700 ml-1 focus:outline-none p-1 rounded-md hover:bg-gray-100"
                              >
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  width="14"
                                  height="14"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  className="feather feather-copy"
                                >
                                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                                <span className="sr-only">Copy parent ID</span>
                              </button>
                            </div>
                          ) : (
                            <div className="flex items-center gap-1">
                              <p className="text-sm" title={selectedSpan.parent_id}>
                                {selectedSpan.parent_id
                                  ? `${selectedSpan.parent_id.substring(0, 8)}...`
                                  : ''}
                              </p>
                              <button
                                onClick={() => {
                                  navigator.clipboard.writeText(selectedSpan.parent_id || '');
                                  toast({
                                    title: 'Copied to clipboard',
                                    description: 'Parent ID has been copied to clipboard',
                                    duration: 2000,
                                  });
                                }}
                                className="text-gray-500 hover:text-gray-700 ml-1 focus:outline-none p-1 rounded-md hover:bg-gray-100"
                              >
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  width="14"
                                  height="14"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  className="feather feather-copy"
                                >
                                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                                </svg>
                                <span className="sr-only">Copy parent ID</span>
                              </button>
                            </div>
                          )}
                        </div>
                        <div></div> {/* Empty div to maintain grid alignment */}
                      </>
                    )}
                  </div>

                  <Tabs defaultValue="all" className="flex-1 flex flex-col min-h-0">
                    <TabsList className="flex-shrink-0 w-full">
                      <TabsTrigger value="all" className="flex-1">
                        All Data
                      </TabsTrigger>
                      <TabsTrigger value="input" className="flex-1">
                        Input
                      </TabsTrigger>
                      <TabsTrigger value="output" className="flex-1">
                        Output
                      </TabsTrigger>
                      <TabsTrigger value="data" className="flex-1">
                        Data
                      </TabsTrigger>
                    </TabsList>

                    <TabsContent
                      value="all"
                      className="flex-1 min-h-0 mt-2 data-[state=active]:flex flex-col"
                    >
                      <div className="border rounded-md bg-muted flex-1 min-h-0 overflow-hidden">
                        <pre className="p-4 text-xs whitespace-pre-wrap h-full overflow-y-auto">
                          {JSON.stringify(
                            {
                              id: selectedSpan.id,
                              trace_id: selectedSpan.trace_id,
                              parent_id: selectedSpan.parent_id,
                              name: selectedSpan.name,
                              start_time: selectedSpan.start_time,
                              end_time: selectedSpan.end_time,
                              input: selectedSpan.input,
                              output: selectedSpan.output,
                              data: selectedSpan.data,
                            },
                            null,
                            2,
                          )}
                        </pre>
                      </div>
                    </TabsContent>

                    <TabsContent
                      value="input"
                      className="flex-1 min-h-0 mt-2 data-[state=active]:flex flex-col"
                    >
                      <div className="border rounded-md bg-muted flex-1 min-h-0 overflow-hidden">
                        <pre className="p-4 text-xs whitespace-pre-wrap h-full overflow-y-auto">
                          {selectedSpan.input
                            ? JSON.stringify(selectedSpan.input, null, 2)
                            : 'No input data'}
                        </pre>
                      </div>
                    </TabsContent>

                    <TabsContent
                      value="output"
                      className="flex-1 min-h-0 mt-2 data-[state=active]:flex flex-col"
                    >
                      <div className="border rounded-md bg-muted flex-1 min-h-0 overflow-hidden">
                        <pre className="p-4 text-xs whitespace-pre-wrap h-full overflow-y-auto">
                          {selectedSpan.output
                            ? JSON.stringify(selectedSpan.output, null, 2)
                            : 'No output data'}
                        </pre>
                      </div>
                    </TabsContent>

                    <TabsContent
                      value="data"
                      className="flex-1 min-h-0 mt-2 data-[state=active]:flex flex-col"
                    >
                      <div className="border rounded-md bg-muted flex-1 min-h-0 overflow-hidden">
                        <pre className="p-4 text-xs whitespace-pre-wrap h-full overflow-y-auto">
                          {selectedSpan.data
                            ? JSON.stringify(selectedSpan.data, null, 2)
                            : 'No data'}
                        </pre>
                      </div>
                    </TabsContent>
                  </Tabs>
                </div>
              </CardContent>
            </>
          ) : (
            <div className="h-full flex items-center justify-center p-8">
              <div className="text-center">
                <div className="text-muted-foreground mb-4">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="40"
                    height="40"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="mx-auto mb-2"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <path d="M8 12h8" />
                  </svg>
                </div>
                <h3 className="text-lg font-medium mb-2">No span selected</h3>
                <p className="text-sm text-muted-foreground max-w-[300px]">
                  Click on a span in the timeline or a label in the left sidebar to view its details
                  here.
                </p>
              </div>
            </div>
          )}
        </Card>
      </div>
    </main>
  );
}
