'use client';

import { Send, X, Loader2, Ellipsis, Check } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { useState } from 'react';
import { AttachmentButton } from './attachment-button';
import { AttachmentList } from './attachment-list';
import { FilePicker } from './file-picker';
import { FileAttachment } from '@/types/messages';

interface TaskControlsProps {
  userInput: string;
  setUserInput: (value: string) => void;
  handleSubmit: (e: React.FormEvent) => void;
  isThinking: boolean;
  taskInTerminalState: boolean;
  handleApproveTask: () => void;
  isApproving: boolean;
  handleCancelTask: () => void;
  isCancelling: boolean;
  onAttachmentsChange?: (
    attachments: FileAttachment[] | ((curr: FileAttachment[]) => FileAttachment[]),
  ) => void;
}

const TaskControls: React.FC<TaskControlsProps> = ({
  userInput,
  setUserInput,
  handleSubmit,
  isThinking,
  taskInTerminalState,
  handleApproveTask,
  isApproving,
  handleCancelTask,
  isCancelling,
  onAttachmentsChange,
}) => {
  const [attachments, setAttachments] = useState<FileAttachment[]>([]);

  // Custom submit handler to check if all attachments are uploaded
  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Check if there are any attachments still uploading
    const hasUploadingAttachments = attachments.some(
      attachment => attachment.status === 'uploading',
    );

    if (hasUploadingAttachments) {
      // Prevent submission if files are still uploading
      alert('Please wait for all attachments to finish uploading');
      return;
    }

    // Get all successfully uploaded attachments
    const successfulAttachments = attachments.filter(
      attachment => attachment.status === 'success' && attachment.file_id,
    );

    console.log('TaskControls passing successful attachments:', successfulAttachments.length);

    // Attach the successful attachments to the event object
    (e as any).attachments = successfulAttachments;

    // Also forward the attachments through the callback for redundancy
    if (onAttachmentsChange) {
      onAttachmentsChange(successfulAttachments);
    }

    // Call the parent's submit handler with our modified event
    handleSubmit(e);

    // Clear attachments after submission
    setAttachments([]);
  };

  // Handle attachment updates
  const handleAttachmentChange = (
    newAttachments: FileAttachment[] | ((curr: FileAttachment[]) => FileAttachment[]),
  ) => {
    // Update local state
    if (typeof newAttachments === 'function') {
      setAttachments(prev => {
        const updatedAttachments = newAttachments(prev);
        // Forward to parent component if available
        if (onAttachmentsChange) {
          onAttachmentsChange(updatedAttachments);
        }
        return updatedAttachments;
      });
    } else {
      setAttachments(newAttachments);
      // Forward to parent component if available
      if (onAttachmentsChange) {
        onAttachmentsChange(newAttachments);
      }
    }
  };

  // Handle attachment removal
  const handleRemoveAttachment = (id: string) => {
    setAttachments(prev => {
      const filtered = prev.filter(attachment => attachment.file_id !== id);
      // Also notify parent of removal
      if (onAttachmentsChange) {
        onAttachmentsChange(filtered);
      }
      return filtered;
    });
  };

  return (
    <div className="max-w-5xl w-full mx-auto px-4 pb-4">
      {attachments.length > 0 && (
        <div className="mb-2">
          <AttachmentList attachments={attachments} onRemove={handleRemoveAttachment} />
        </div>
      )}
      <form onSubmit={onSubmit} className="mb-6 w-full">
        <div className="flex w-full">
          <div className="flex-grow flex flex-col rounded-md border bg-white overflow-hidden">
            <Input
              type="text"
              placeholder="Provide new instructions..."
              value={userInput}
              onChange={e => setUserInput(e.target.value)}
              className="flex-grow border-0 rounded-t-md px-4 py-4 text-base min-h-[52px] placeholder:text-gray-400 placeholder:font-light placeholder:opacity-70 focus-visible:ring-0 focus-visible:ring-offset-0"
              disabled={taskInTerminalState || isThinking}
            />
            <div className="flex items-center px-3 py-2 border-t bg-gray-50">
              <div className="flex items-center space-x-3">
                <FilePicker
                  onFileSelect={handleAttachmentChange}
                  currentAttachments={attachments}
                  disabled={taskInTerminalState || isThinking}
                />
                <AttachmentButton
                  onAttachmentChange={handleAttachmentChange}
                  disabled={taskInTerminalState || isThinking}
                  currentAttachments={attachments}
                />
              </div>
              <div className="ml-auto flex space-x-2">
                <Button
                  type="submit"
                  size="icon"
                  variant="ghost"
                  className="text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-full w-9 h-9"
                  disabled={taskInTerminalState || isThinking}
                >
                  {isThinking ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                  <span className="sr-only">Send</span>
                </Button>
                <Button
                  size="sm"
                  className="bg-green-600 hover:bg-green-700"
                  onClick={handleApproveTask}
                  disabled={taskInTerminalState || isApproving}
                >
                  {isApproving ? (
                    <Ellipsis className="h-4 w-4 mr-2 animate-pulse" />
                  ) : (
                    <>
                      <Check className="h-4 w-4 mr-2" />
                      <span>Done</span>
                    </>
                  )}
                </Button>
                <Button
                  size="sm"
                  className="bg-red-500 hover:bg-red-600"
                  onClick={handleCancelTask}
                  disabled={taskInTerminalState || isCancelling}
                >
                  {isCancelling ? (
                    <Ellipsis className="h-4 w-4 mr-2 animate-pulse" />
                  ) : (
                    <>
                      <X className="h-4 w-4 mr-2" />
                      <span>Cancel</span>
                    </>
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
};

export default TaskControls;
