/**
 * Right-rail agent config form.
 *
 * React Hook Form + Zod schema validates client-side; backend Pydantic
 * remains authoritative on save. The tool checklist consumes
 * `GET /v1/tools` (cached for the FastAPI process lifetime, refreshed
 * here via TanStack Query with `staleTime: Infinity`).
 *
 * Pattern: form has its own internal state. On "Apply", the form pushes
 * the patched agent back into the canvas's working topology via the
 * `onApply` prop. The canvas owns the persisted workflow definition;
 * this form is editing a slice of it.
 */
import { zodResolver } from "@hookform/resolvers/zod";
import { useQuery } from "@tanstack/react-query";
import { useEffect, type ReactElement } from "react";
import { Controller, useForm } from "react-hook-form";
import { z } from "zod";

import { listTools, type AgentSpec } from "../../../lib/api";
import { useCapabilities } from "../../../providers/capabilities";

import "./AgentConfigForm.css";

const MEMORY_OPTIONS = [
  { id: "single_run", label: "single run" },
  { id: "session", label: "session" },
  { id: "persistent", label: "persistent" },
] as const;

const schema = z.object({
  name: z.string().min(1, "name required"),
  goal: z.string(),
  instruction: z.string().min(1, "instruction required"),
  model_name: z.string().min(1, "model required"),
  provider: z.enum([
    "openai",
    "openrouter",
    "google",
    "anthropic",
    "xai",
    "openai-oauth",
    "anthropic-oauth",
  ]),
  temperature: z.coerce.number().min(0).max(2),
  max_tokens: z.coerce.number().int().positive(),
  memory_retention: z.enum(["single_run", "session", "persistent"]),
  tools: z.array(z.string()),
});

type FormValues = z.infer<typeof schema>;

interface AgentConfigFormProps {
  agentId: string;
  agent: AgentSpec;
  onApply: (next: AgentSpec) => void;
  onDelete: () => void;
}

export function AgentConfigForm({
  agentId,
  agent,
  onApply,
  onDelete,
}: AgentConfigFormProps): ReactElement {
  const { token } = useCapabilities();

  const toolsQuery = useQuery({
    queryKey: ["tools"],
    queryFn: () => listTools(token ?? ""),
    enabled: Boolean(token),
    staleTime: Infinity,
  });

  const defaultValues: FormValues = {
    name: agent.name,
    goal: agent.goal ?? "",
    instruction: agent.instruction,
    model_name: agent.agent_model.name,
    provider: (agent.agent_model.provider ?? "anthropic") as FormValues["provider"],
    temperature: agent.agent_model.temperature ?? 0.7,
    max_tokens: agent.agent_model.max_tokens ?? 8192,
    memory_retention: (agent.memory_retention ?? "session") as FormValues["memory_retention"],
    tools: agent.tools ?? [],
  };

  const {
    control,
    handleSubmit,
    register,
    reset,
    watch,
    formState: { errors, isDirty },
  } = useForm<FormValues>({
    defaultValues,
    resolver: zodResolver(schema),
  });

  // Reset form when the selected agent changes.
  useEffect(() => {
    reset(defaultValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agentId]);

  function onSubmit(values: FormValues) {
    const next: AgentSpec = {
      ...agent,
      name: values.name,
      goal: values.goal,
      instruction: values.instruction,
      memory_retention: values.memory_retention,
      tools: values.tools,
      agent_model: {
        ...agent.agent_model,
        name: values.model_name,
        provider: values.provider,
        temperature: values.temperature,
        max_tokens: values.max_tokens,
      },
    };
    onApply(next);
    reset(values, { keepValues: true });
  }

  const tools = watch("tools");

  return (
    <form
      className="agent-form"
      onSubmit={handleSubmit(onSubmit)}
      data-testid="agent-form"
    >
      <header className="agent-form-tag">
        &lt;agent name=&quot;{watch("name")}&quot; model=&quot;{watch("model_name")}&quot;
        tools=&#123;{tools.join(",")}&#125; /&gt;
      </header>

      <section className="agent-form-section">
        <h3>Identity</h3>
        <label className="agent-form-field">
          <span>Name *</span>
          <input
            type="text"
            {...register("name")}
            data-testid="agent-form-name"
          />
          {errors.name ? <em className="agent-form-error">{errors.name.message}</em> : null}
        </label>
        <label className="agent-form-field">
          <span>Goal</span>
          <input type="text" {...register("goal")} />
        </label>
      </section>

      <section className="agent-form-section">
        <h3>Model</h3>
        <label className="agent-form-field">
          <span>Provider</span>
          <select {...register("provider")}>
            <option value="anthropic">anthropic</option>
            <option value="openai">openai</option>
            <option value="openrouter">openrouter</option>
            <option value="google">google</option>
            <option value="xai">xai</option>
            <option value="openai-oauth">openai-oauth</option>
            <option value="anthropic-oauth">anthropic-oauth</option>
          </select>
        </label>
        <label className="agent-form-field">
          <span>Model name *</span>
          <input
            type="text"
            {...register("model_name")}
            placeholder="e.g. claude-opus-4-7"
            data-testid="agent-form-model"
          />
          {errors.model_name ? <em className="agent-form-error">{errors.model_name.message}</em> : null}
        </label>
        <div className="agent-form-row">
          <label className="agent-form-field">
            <span>Temperature</span>
            <input
              type="number"
              step="0.1"
              {...register("temperature")}
            />
          </label>
          <label className="agent-form-field">
            <span>Max tokens</span>
            <input
              type="number"
              step="64"
              {...register("max_tokens")}
            />
          </label>
        </div>
      </section>

      <section className="agent-form-section">
        <h3>Instruction *</h3>
        <textarea
          {...register("instruction")}
          rows={5}
          data-testid="agent-form-instruction"
          placeholder="You are an agent. Find sources on the user's topic. Cite each one."
        />
        {errors.instruction ? (
          <em className="agent-form-error">{errors.instruction.message}</em>
        ) : null}
      </section>

      <section className="agent-form-section">
        <h3>Tools</h3>
        {toolsQuery.isError ? (
          <p className="agent-form-error">Couldn't load tools.</p>
        ) : toolsQuery.isLoading ? (
          <p className="agent-form-loading">Loading tools…</p>
        ) : (
          <Controller
            name="tools"
            control={control}
            render={({ field }) => (
              <div className="agent-form-tools">
                {(toolsQuery.data?.items ?? []).map((tool) => {
                  const checked = field.value.includes(tool.name);
                  return (
                    <label key={tool.name} className="agent-form-tool">
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={(e) => {
                          if (e.target.checked) field.onChange([...field.value, tool.name]);
                          else field.onChange(field.value.filter((t) => t !== tool.name));
                        }}
                        data-testid={`agent-form-tool-${tool.name}`}
                      />
                      <span className="agent-form-tool-name">{tool.name}</span>
                      {tool.description ? (
                        <span className="agent-form-tool-desc">{tool.description}</span>
                      ) : null}
                    </label>
                  );
                })}
              </div>
            )}
          />
        )}
      </section>

      <section className="agent-form-section">
        <h3>Memory</h3>
        <label className="agent-form-field">
          <span>Retention</span>
          <select {...register("memory_retention")}>
            {MEMORY_OPTIONS.map((option) => (
              <option key={option.id} value={option.id}>{option.label}</option>
            ))}
          </select>
        </label>
      </section>

      <footer className="agent-form-actions">
        <button
          type="button"
          className="agent-form-delete"
          onClick={onDelete}
          data-testid="agent-form-delete"
        >
          Delete node
        </button>
        <button
          type="submit"
          className="agent-form-apply"
          disabled={!isDirty}
          data-testid="agent-form-apply"
        >
          Apply
        </button>
      </footer>
    </form>
  );
}
