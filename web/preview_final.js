import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function viewUrl(media) {
  const params = new URLSearchParams({
    filename: media.filename,
    type: media.type || "output",
    subfolder: media.subfolder || "",
  });
  return api.apiURL(`/view?${params.toString()}`);
}

app.registerExtension({
  name: "ace_h3.PreviewFinalMusicVideo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "PreviewFinalMusicVideo") {
      return;
    }

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);

      const media =
        message?.gifs?.[0] ||
        (message?.animated?.[0] ? message?.images?.[0] : null) ||
        message?.images?.find?.((x) => String(x.filename || "").toLowerCase().endsWith(".mp4"));

      if (!media?.filename) {
        return;
      }

      if (this.__acePreviewVideo) {
        try {
          this.__acePreviewVideo.remove();
        } catch (_) {}
        this.__acePreviewVideo = null;
      }
      if (this.__aceVideoWidget) {
        const idx = this.widgets?.indexOf(this.__aceVideoWidget);
        if (idx >= 0) {
          this.widgets.splice(idx, 1);
        }
        this.__aceVideoWidget = null;
      }

      const video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.style.width = "100%";
      video.style.maxHeight = "360px";
      video.style.background = "#111";
      video.src = viewUrl(media);

      this.__acePreviewVideo = video;
      this.__aceVideoWidget = this.addDOMWidget("final_video", "FINAL_VIDEO", video, {
        serialize: false,
        hideOnZoom: false,
      });
      this.setSize([Math.max(this.size[0], 420), Math.max(this.size[1], 320)]);
      this.setDirtyCanvas?.(true, true);
    };
  },
});
