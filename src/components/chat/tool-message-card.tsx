import { useState } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { motion } from 'framer-motion';
import { Badge } from '@/components/ui/badge';
import { ChevronDown, ChevronRight, Wrench, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface ToolMessageCardProps {
  type: 'request' | 'response';
  name: string;
  content: any;
  style?: 'static' | 'active';
}

function ToolMessageCard({ type, name, content, style }: ToolMessageCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isVisible, setIsVisible] = useState(false);

  // Make the card visible immediately when it receives content
  if (!content) return null;

  const Icon = type === 'request' ? Wrench : CheckCircle;
  const iconColor = type === 'request' ? 'text-blue-500' : 'text-green-500';
  const title = type === 'request' 
    ? `Using tool: ${name}`
    : `Tool response: ${name}`;

  const formattedContent = type === 'request'
    ? JSON.stringify(content, null, 2)
    : typeof content === 'string' ? content : JSON.stringify(content, null, 2);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="relative flex"
    >
      <Card
        onClick={() => setIsExpanded(!isExpanded)}
        className={cn(
          "w-full max-w-full rounded-md overflow-hidden text-gray-800",
          "bg-white shadow-none border-gray-200",
          style === 'active' && "border-l-4 border-l-blue-500",
          "cursor-pointer hover:bg-gray-50 transition-colors"
        )}
      >
        <div className="flex flex-col justify-between p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Icon className={cn("h-4 w-4", iconColor)} />
              <span className="text-sm font-medium">{title}</span>
            </div>
            <div className="h-6 w-6 flex items-center justify-center">
              {isExpanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </div>
          </div>
          
          {isExpanded && (
            <CardContent className="p-0 mt-2">
              <pre className={cn(
                "text-sm bg-gray-50 p-3 rounded-md overflow-x-auto",
                type === 'request' ? "font-mono" : "whitespace-pre-wrap"
              )}>
                {formattedContent}
              </pre>
            </CardContent>
          )}
        </div>
      </Card>
    </motion.div>
  );
}

export default ToolMessageCard; 