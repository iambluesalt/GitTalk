import { type RouteConfig, index, layout, route } from "@react-router/dev/routes";

export default [
  layout("components/AppLayout.tsx", [
    index("routes/home.tsx"),
    route("projects", "routes/projects.tsx"),
    route("clone", "routes/clone.tsx"),
    route("projects/:projectId", "routes/project-detail.tsx"),
    route("chat/:projectId", "routes/chat.tsx"),
    route("settings", "routes/settings.tsx"),
    route("guide", "routes/guide.tsx"),
  ]),
] satisfies RouteConfig;
