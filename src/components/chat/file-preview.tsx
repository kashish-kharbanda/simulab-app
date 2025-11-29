'use client';

import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Download, FileText, Image as ImageIcon, FileIcon } from 'lucide-react';

interface FilePreviewProps {
  fileId: string;
  fileName: string;
  fileType: string;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export function FilePreview({
  fileId,
  fileName,
  fileType,
  isOpen,
  onOpenChange,
}: FilePreviewProps) {
  // Handle direct file opening in browser
  const handleOpenFile = () => {
    window.open(`/api/file/${fileId}`, '_blank');
    onOpenChange(false); // Close the dialog after opening
  };

  // Handle file download
  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = `/api/file/${fileId}`;
    a.download = fileName; // This will prompt download instead of opening
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    onOpenChange(false); // Close the dialog after download starts
  };

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {fileType.startsWith('image/') ? (
              <ImageIcon className="h-5 w-5" />
            ) : fileType === 'application/pdf' ? (
              <FileText className="h-5 w-5" />
            ) : (
              <FileIcon className="h-5 w-5" />
            )}
            <span className="truncate">{fileName}</span>
          </DialogTitle>
        </DialogHeader>

        <div className="flex justify-center gap-4 py-6">
          <Button onClick={handleOpenFile} className="flex items-center gap-2">
            Open in Browser
          </Button>
          <Button onClick={handleDownload} variant="outline" className="flex items-center gap-2">
            <Download className="h-4 w-4" />
            Download
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
