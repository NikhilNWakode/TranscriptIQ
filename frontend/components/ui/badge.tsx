import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium whitespace-nowrap [&_svg]:size-3",
  {
    variants: {
      variant: {
        default: "border-transparent bg-primary text-primary-foreground",
        secondary: "border-transparent bg-secondary text-secondary-foreground",
        outline: "text-foreground bg-card",
        muted: "border-transparent bg-muted text-muted-foreground",
        common: "border-emerald-200 bg-emerald-50 text-emerald-700",
        emphasis: "border-amber-200 bg-amber-50 text-amber-800",
        disagree: "border-rose-200 bg-rose-50 text-rose-700",
        ai: "border-ai-border bg-ai text-ai-foreground",
        source: "border-source-border bg-source text-source-foreground",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement>, VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
