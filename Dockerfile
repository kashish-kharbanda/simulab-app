FROM node:20-alpine AS base
RUN npm install -g turbo pnpm
 
FROM base AS pruner
WORKDIR /app
COPY . .
RUN turbo prune web --docker
 
# Add lockfile and package.json's of isolated subworkspace
FROM base AS installer
WORKDIR /app
COPY --from=pruner /app/out/json/ .
COPY --from=pruner /app/out/pnpm-lock.yaml ./pnpm-lock.yaml
COPY --from=pruner /app/turbo.json ./turbo.json
RUN pnpm install --frozen-lockfile
 
FROM base AS builder
WORKDIR /app
COPY --from=installer /app/ . 
COPY --from=pruner /app/out/full/ .
RUN pnpm run build --env-mode=loose
 
FROM base AS runner
WORKDIR /app
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs -G nodejs
USER nextjs
 
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/public ./apps/web/public

CMD node apps/web/server.js