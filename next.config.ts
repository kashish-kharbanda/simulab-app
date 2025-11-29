import type { NextConfig } from 'next';
import path from 'node:path';

const nextConfig: NextConfig = {
  // Set to current directory to avoid monorepo detection issues
  // This tells Next.js to treat this directory as the root for file tracing
  outputFileTracingRoot: path.join(__dirname),
  output: 'standalone',
};

export default nextConfig;
