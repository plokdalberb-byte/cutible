const { CutibleClient } = require("../client");

// Mock axios
jest.mock("axios", () => {
  const mockAxios = {
    create: jest.fn(() => mockAxios),
    get: jest.fn(() => Promise.resolve({ data: {} })),
    post: jest.fn(() => Promise.resolve({ data: {} })),
  };
  return mockAxios;
});

describe("CutibleClient", () => {
  let client;

  beforeEach(() => {
    client = new CutibleClient({ apiUrl: "http://localhost:8000" });
    jest.clearAllMocks();
  });

  describe("constructor", () => {
    it("creates client with apiUrl", () => {
      expect(client.apiUrl).toBe("http://localhost:8000");
      expect(client.http).toBeTruthy();
    });

    it("creates client without apiUrl", () => {
      const c = new CutibleClient();
      expect(c.apiUrl).toBeNull();
      expect(c.http).toBeNull();
    });
  });

  describe("HTTP mode methods", () => {
    it("createProject calls POST /projects", async () => {
      const axios = require("axios");
      axios.post.mockResolvedValue({
        data: { created: "test", summary: { id: "test" } },
      });

      const result = await client.createProject("test", { fps: 30 });
      expect(axios.post).toHaveBeenCalledWith(
        "/projects",
        expect.objectContaining({ id: "test", fps: 30 })
      );
      expect(result.created).toBe("test");
    });

    it("addAsset calls POST /projects/:id/verbs", async () => {
      const axios = require("axios");
      client.projectId = "test";
      axios.post.mockResolvedValue({ data: { verb: "add_asset", changed: [] } });

      await client.addAsset("a1", "video", { uri: "test.mp4" });
      expect(axios.post).toHaveBeenCalledWith(
        "/projects/test/verbs",
        expect.objectContaining({ verb: "add_asset" })
      );
    });

    it("addClip calls POST with correct args", async () => {
      const axios = require("axios");
      client.projectId = "test";
      axios.post.mockResolvedValue({ data: { verb: "add_clip", changed: [] } });

      await client.addClip("v1", "a1", { srcIn: 0, srcOut: 10 });
      expect(axios.post).toHaveBeenCalledWith(
        "/projects/test/verbs",
        expect.objectContaining({
          verb: "add_clip",
          args: expect.objectContaining({ asset: "a1" }),
        })
      );
    });

    it("render calls POST /projects/:id/render", async () => {
      const axios = require("axios");
      client.projectId = "test";
      axios.post.mockResolvedValue({ data: { output: "out.mp4" } });

      const result = await client.render("out.mp4");
      expect(axios.post).toHaveBeenCalledWith(
        "/projects/test/render",
        expect.objectContaining({ output: "out.mp4" })
      );
    });

    it("runAgent calls POST /agent/run", async () => {
      const axios = require("axios");
      axios.post.mockResolvedValue({ data: { passed: true } });

      const result = await client.runAgent("make a recap", {
        targetDuration: 30,
        style: "energetic",
      });
      expect(axios.post).toHaveBeenCalledWith(
        "/agent/run",
        expect.objectContaining({ brief: "make a recap" })
      );
    });

    it("ingestAsset calls POST /ingest", async () => {
      const axios = require("axios");
      axios.post.mockResolvedValue({ data: { success: true } });

      await client.ingestAsset("a1", "/path/to/video.mp4");
      expect(axios.post).toHaveBeenCalledWith(
        "/ingest",
        expect.objectContaining({ asset_id: "a1" })
      );
    });

    it("exportOtio calls GET with output_path", async () => {
      const axios = require("axios");
      client.projectId = "test";
      axios.get.mockResolvedValue({ data: { ok: true } });

      await client.exportOtio("output.otio");
      expect(axios.get).toHaveBeenCalledWith(
        "/projects/test/otio",
        expect.objectContaining({ params: { output_path: "output.otio" } })
      );
    });

    it("searchIndex calls GET /index/search", async () => {
      const axios = require("axios");
      axios.get.mockResolvedValue({ data: { results: [{ text: "hello" }] } });

      const results = await client.searchIndex("hello");
      expect(axios.get).toHaveBeenCalledWith(
        "/index/search",
        expect.objectContaining({ params: { q: "hello" } })
      );
    });
  });

  describe("error handling", () => {
    it("throws when no apiUrl for createProject", async () => {
      const c = new CutibleClient();
      await expect(c.createProject("test")).rejects.toThrow("HTTP mode required");
    });

    it("throws when no projectId for addClip", async () => {
      await expect(client.addClip("v1", "a1")).rejects.toThrow("HTTP mode required");
    });
  });
});
