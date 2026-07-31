/**
 * Surface flags — which routes this deployment exposes.
 *
 * The product is **ViTreous**: the workbench at `/` and its malignancy lens at
 * `/lens`. The UMT-ViT line (`/umtvit`, `/sgp`) is a separate self-supervised
 * topography experiment — genuinely interesting research, but not part of the
 * thing a user is meant to open. Shipping both invites a visitor to wander into
 * a SOM lattice viewer and conclude that *is* the product.
 *
 * Hidden, not deleted. The experiment code, its bundles and its 150 tests all
 * stay in the repo; set `NEXT_PUBLIC_VITREOUS_EXPERIMENTS=1` and rebuild to get
 * the routes and their nav links back. Note NEXT_PUBLIC_* is inlined at BUILD
 * time, so flipping this on a running deployment requires a redeploy.
 */

export function experimentsEnabled(): boolean {
  return process.env.NEXT_PUBLIC_VITREOUS_EXPERIMENTS === "1";
}
