'use client';

import { FC, useEffect, useRef, useState } from 'react';
import { Crepe } from '@milkdown/crepe';
import '@milkdown/crepe/theme/common/style.css';
import '@milkdown/crepe/theme/nord.css';
import styles from './milkdown.module.css';

interface EditorProps {
  markdown: string;
  readOnly?: boolean;
  onChange?: (markdown: string) => void;
  className?: string;
  onReady?: (ready: boolean) => void;
}

const DEFAULT_FEATURES = {
  [Crepe.Feature.BlockEdit]: false, // Disable block handle and menu
};

/**
 * A Markdown editor component using Milkdown Crepe
 */
const MarkdownEditor: FC<EditorProps> = (props: EditorProps) => {
  const { markdown, readOnly = false, onChange, className, onReady } = props;
  const editorRef = useRef<Crepe | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isEditorReady, setIsEditorReady] = useState(false);

  // Initialize the editor
  useEffect(() => {
    if (!containerRef.current) return;

    // Create the Crepe editor with disabled block handle
    const crepe = new Crepe({
      root: containerRef.current,
      defaultValue: markdown,
      features: DEFAULT_FEATURES,
    });

    editorRef.current = crepe;

    // Create the editor
    crepe.create().then(() => {
      console.log('Editor created');
      
      // Set readonly mode if needed
      if (readOnly) {
        crepe.setReadonly(true);
      }

      // Add listener for markdown updates
      if (onChange) {
        crepe.on((listener) => {
          listener.markdownUpdated((markdown) => {
            onChange(markdown);
          });
        });
      }

      // Mark editor as ready
      setIsEditorReady(true);
      onReady?.(true);
    });

    // Cleanup function
    return () => {
      if (editorRef.current) {
        editorRef.current.destroy();
        editorRef.current = null;
      }
      setIsEditorReady(false);
      onReady?.(false);
    };
  }, []);

  // Update editor content when markdown prop changes
  useEffect(() => {
    // Only update if the editor is ready and we have a reference to it
    if (isEditorReady && editorRef.current) {
      // We need to destroy and recreate the editor to update content
      // This is a limitation of the current implementation
      editorRef.current.destroy();
      onReady?.(false);
      
      const crepe = new Crepe({
        root: containerRef.current!,
        defaultValue: markdown,
        features: DEFAULT_FEATURES,
      });
      
      editorRef.current = crepe;
      
      crepe.create().then(() => {
        if (readOnly) {
          crepe.setReadonly(true);
        }
        
        if (onChange) {
          crepe.on((listener) => {
            listener.markdownUpdated((markdown) => {
              onChange(markdown);
            });
          });
        }
        onReady?.(true);
      });
    }
  }, [markdown, readOnly, onChange, isEditorReady]);

  return (
    <div className={`${styles.wrapper} ${className || ''}`}>
      <div ref={containerRef} />
    </div>
  );
};

export default MarkdownEditor;
