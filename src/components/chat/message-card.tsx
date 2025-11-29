import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';
import MarkdownEditor from '../editor/markdown-editor';
import { Badge } from '@/components/ui/badge';
import { FileAttachment } from '@/types/messages';
import { FilePreview } from './file-preview';
import { FileIcon, ExternalLink } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface MessageCardProps {
  content: string;
  style?: 'static' | 'active';
  role: 'user' | 'agent';
  attachments?: FileAttachment[];
}

function MessageCard({ content, role, style, attachments }: MessageCardProps) {
  const [previewFile, setPreviewFile] = useState<{
    id: string;
    name: string;
    type: string;
    file_id?: string;
  } | null>(null);
  const [isMarkdownReady, setIsMarkdownReady] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  // Make the card visible immediately when it receives content
  useEffect(() => {
    if (content) {
      setIsVisible(true);
    }
  }, [content]);

  // Don't render anything if there's no content
  if (!content) return null;

  const handleAttachmentClick = (attachment: FileAttachment) => {
    if (attachment.status === 'success' && attachment.file_id) {
      // Open directly in browser
      window.open(`/api/file/${attachment.file_id}`, '_blank');
    }
  };

  const handleShowOptions = (e: React.MouseEvent, attachment: FileAttachment) => {
    e.stopPropagation(); // Prevent opening in new tab

    if (attachment.status === 'success' && attachment.file_id) {
      setPreviewFile({
        id: attachment.file_id,
        name: attachment.name,
        type: attachment.type,
        file_id: attachment.file_id,
      });
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: isVisible ? 1 : 0, y: isVisible ? 0 : -20 }}
      transition={{ duration: 0.8, ease: "easeInOut" }}
      className="relative flex"
    >
      <Card
        className={`w-full max-w-full rounded-md overflow-hidden text-gray-800
          ${role === 'user' ? 'bg-gray-100' : 'bg-white'}
          shadow-none border-gray-200`}
      >
        {style === 'active' && (
          <div className="animate-pulse bg-gradient-to-r from-blue-400 via-teal-500 to-purple-500 h-1"></div>
        )}
        <div className="flex flex-col justify-between p-6">
          <Badge variant="outline" className={`w-fit text-sm flex items-center gap-1 mb-4`}>
            {role === 'user' ? 'User' : 'Agent'}
          </Badge>
          <CardContent className="p-0">
            <MarkdownEditor 
              readOnly 
              markdown={content} 
              className="!p-0"
              onReady={setIsMarkdownReady}
            />
            {attachments && attachments.length > 0 && (
              <div className="mt-4">
                {attachments.map(attachment => (
                  <div
                    key={attachment.file_id}
                    className="flex items-center justify-between p-2 rounded-md border mb-2 bg-gray-50 hover:bg-gray-100 cursor-pointer"
                    onClick={() => handleAttachmentClick(attachment)}
                  >
                    <div className="flex items-center space-x-2 overflow-hidden">
                      <FileIcon className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
                      <div className="flex-1">
                        <p className="text-sm font-medium truncate">{attachment.name}</p>
                        <p className="text-xs text-gray-500">
                          {formatFileSize(attachment.size)} • Click to open
                        </p>
                      </div>
                    </div>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 rounded-full"
                      onClick={e => handleShowOptions(e, attachment)}
                      aria-label="Show download options"
                    >
                      <ExternalLink className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </div>
      </Card>

      {/* File options dialog */}
      {previewFile && previewFile.file_id && (
        <FilePreview
          fileId={previewFile.file_id}
          fileName={previewFile.name}
          fileType={previewFile.type}
          isOpen={!!previewFile}
          onOpenChange={open => {
            if (!open) setPreviewFile(null);
          }}
        />
      )}
    </motion.div>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export default MessageCard;
