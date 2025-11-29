'use client';

import React, { createContext, useContext, useState, useEffect } from 'react';
import { agentService } from '@/services';
import type { Task } from '@/types/tasks';

interface TasksContextType {
  tasks: Task[];
  selectedTask: Task | null;
  setSelectedTask: (task: Task | null) => void;
  refreshTasks: () => Promise<void>;
  isLoading: boolean;
}

const TasksContext = createContext<TasksContextType>({
  tasks: [],
  selectedTask: null,
  setSelectedTask: () => {},
  refreshTasks: async () => {},
  isLoading: true,
});

export function TasksProvider({ children }: { children: React.ReactNode }) {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const refreshTasks = async () => {
    try {
      const response = await agentService.listTasks();
      setTasks(response);
    } catch (error) {
      console.error('Error fetching tasks:', error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    refreshTasks();
    
    // Set up polling for task updates
    const interval = setInterval(refreshTasks, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <TasksContext.Provider
      value={{
        tasks,
        selectedTask,
        setSelectedTask,
        refreshTasks,
        isLoading,
      }}
    >
      {children}
    </TasksContext.Provider>
  );
}

export const useTasks = () => useContext(TasksContext); 