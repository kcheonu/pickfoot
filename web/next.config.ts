import type { NextConfig } from "next";

const apiOrigin = process.env.PICKFOOT_API_ORIGIN ?? "http://127.0.0.1:8000";
const parsedApiOrigin = new URL(apiOrigin);

if (
  parsedApiOrigin.username
  || parsedApiOrigin.password
  || parsedApiOrigin.pathname !== "/"
  || parsedApiOrigin.search
  || parsedApiOrigin.hash
) {
  throw new Error("PICKFOOT_API_ORIGIN must be an origin without credentials, path, query, or fragment.");
}
if (process.env.VERCEL === "1" && parsedApiOrigin.protocol !== "https:") {
  throw new Error("Vercel deployments require an HTTPS PICKFOOT_API_ORIGIN.");
}

const nextConfig: NextConfig = {
  poweredByHeader: false,
  async rewrites() {
    return [{
      source: "/api/v1/:path*",
      destination: `${parsedApiOrigin.origin}/api/v1/:path*`,
    }];
  },
};

export default nextConfig;
