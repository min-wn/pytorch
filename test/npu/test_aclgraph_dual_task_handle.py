import unittest

import torch

import torch_npu


class TestAclgraphDualTaskHandle(unittest.TestCase):

    def test_dual_task_group_handle_api_export(self):
        self.assertTrue(callable(torch.npu.make_dual_task_group_handle))
        self.assertIs(
            torch.npu.make_dual_task_group_handle,
            torch_npu.npu.graphs.make_dual_task_group_handle)
        self.assertIn("make_dual_task_group_handle", torch_npu.npu.graphs.__all__)

    def test_dual_stream_sync_api_export(self):
        self.assertTrue(hasattr(torch_npu._C, "_DualStreamSyncHandle"))
        self.assertTrue(callable(torch.npu.dual_stream_sync_begin))
        self.assertTrue(callable(torch.npu.dual_stream_sync_end))
        self.assertIs(
            torch.npu.dual_stream_sync_begin,
            torch_npu.npu.graphs.dual_stream_sync_begin)
        self.assertIs(
            torch.npu.dual_stream_sync_end,
            torch_npu.npu.graphs.dual_stream_sync_end)
        self.assertIn("dual_stream_sync_begin", torch_npu.npu.graphs.__all__)
        self.assertIn("dual_stream_sync_end", torch_npu.npu.graphs.__all__)

    def test_dual_fused_infer_attention_score_api_export(self):
        self.assertTrue(hasattr(torch_npu._C, "_dual_fused_infer_attention_score"))
        self.assertTrue(callable(torch.npu.dual_fused_infer_attention_score))
        self.assertTrue(callable(torch_npu.npu_dual_fused_infer_attention_score))
        self.assertIs(
            torch.npu.dual_fused_infer_attention_score,
            torch_npu.npu.graphs.dual_fused_infer_attention_score)
        self.assertIs(
            torch_npu.npu_dual_fused_infer_attention_score,
            torch_npu.npu.graphs.dual_fused_infer_attention_score)
        self.assertIn("dual_fused_infer_attention_score",
                      torch_npu.npu.graphs.__all__)

    def test_dual_fused_infer_attention_score_update_api_export(self):
        self.assertTrue(hasattr(
            torch_npu._C, "_dual_fused_infer_attention_score_update"))
        self.assertTrue(callable(
            torch.npu.dual_fused_infer_attention_score_update))
        self.assertTrue(callable(
            torch_npu.npu_dual_fused_infer_attention_score_update))
        self.assertIs(
            torch.npu.dual_fused_infer_attention_score_update,
            torch_npu.npu.graphs.dual_fused_infer_attention_score_update)
        self.assertIs(
            torch_npu.npu_dual_fused_infer_attention_score_update,
            torch_npu.npu.graphs.dual_fused_infer_attention_score_update)
        self.assertIn("dual_fused_infer_attention_score_update",
                      torch_npu.npu.graphs.__all__)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
    def test_dual_stream_sync_enqueue(self):
        main_stream = torch.npu.current_stream()
        primary_stream = torch.npu.Stream()
        secondary_stream = torch.npu.Stream()

        primary_input = torch.ones(8, device="npu")
        secondary_input = torch.ones(8, device="npu")

        handle = torch.npu.dual_stream_sync_begin(
            main_stream, primary_stream, secondary_stream)
        with torch.npu.stream(primary_stream):
            primary_output = primary_input + 1
        with torch.npu.stream(secondary_stream):
            secondary_output = secondary_input + 2
        torch.npu.dual_stream_sync_end(
            main_stream, primary_stream, secondary_stream, handle)
        torch.npu.synchronize()

        self.assertEqual(float(primary_output.cpu().sum()), 16.0)
        self.assertEqual(float(secondary_output.cpu().sum()), 24.0)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
    def test_dual_fused_infer_attention_score_tnd_matches_two_out_calls(self):
        torch.npu.set_device(0)
        split_tokens = 2
        total_tokens = split_tokens * 2
        kv_tokens = 4
        num_heads = 2
        head_dim = 128
        scale = 1.0

        query = torch.randn(total_tokens, num_heads, head_dim,
                            dtype=torch.bfloat16, device="npu")
        key = torch.randn(kv_tokens, num_heads, head_dim,
                          dtype=torch.bfloat16, device="npu")
        value = torch.randn(kv_tokens, num_heads, head_dim,
                            dtype=torch.bfloat16, device="npu")
        dual_out = torch.empty_like(query)
        ref_out = torch.empty_like(query)
        lse0 = torch.empty((0,), dtype=torch.float32, device="npu")
        lse1 = torch.empty((0,), dtype=torch.float32, device="npu")
        ref_lse0 = torch.empty((0,), dtype=torch.float32, device="npu")
        ref_lse1 = torch.empty((0,), dtype=torch.float32, device="npu")

        torch_npu.npu_fused_infer_attention_score.out(
            query=query[:split_tokens],
            key=key,
            value=value,
            input_layout="TND",
            actual_seq_lengths=[split_tokens],
            actual_seq_lengths_kv=[kv_tokens],
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False,
            out=[ref_out[:split_tokens], ref_lse0])
        torch_npu.npu_fused_infer_attention_score.out(
            query=query[split_tokens:],
            key=key,
            value=value,
            input_layout="TND",
            actual_seq_lengths=[split_tokens],
            actual_seq_lengths_kv=[kv_tokens],
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False,
            out=[ref_out[split_tokens:], ref_lse1])

        main_stream = torch.npu.current_stream()
        primary_stream = torch.npu.Stream()
        secondary_stream = torch.npu.Stream()
        torch.npu.dual_fused_infer_attention_score(
            main_stream,
            primary_stream,
            secondary_stream,
            query,
            key,
            value,
            dual_out,
            actual_seq_lengths_0=[split_tokens],
            actual_seq_lengths_1=[split_tokens],
            actual_seq_lengths_kv_0=[kv_tokens],
            actual_seq_lengths_kv_1=[kv_tokens],
            split_start_0=0,
            split_graph_tokens_0=split_tokens,
            split_start_1=split_tokens,
            split_graph_tokens_1=split_tokens,
            softmax_lse_0=lse0,
            softmax_lse_1=lse1,
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
            input_layout="TND",
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False)
        torch.npu.synchronize()

        torch.testing.assert_close(
            dual_out.cpu(), ref_out.cpu(), rtol=1e-2, atol=1e-2)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
    def test_dual_fused_infer_attention_score_tnd_pa_block_table_matches_two_out_calls(self):
        torch.npu.set_device(0)
        split_tokens = 1
        total_tokens = split_tokens * 2
        num_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 2
        scale = 1.0

        query = torch.randn(total_tokens, num_heads, head_dim,
                            dtype=torch.bfloat16, device="npu")
        key = torch.randn(num_blocks, block_size, num_heads * head_dim,
                          dtype=torch.bfloat16, device="npu")
        value = torch.randn(num_blocks, block_size, num_heads * head_dim,
                            dtype=torch.bfloat16, device="npu")
        block_table = torch.tensor([[0], [1]], dtype=torch.int32, device="npu")
        atten_mask = torch.zeros((2048, 2048), dtype=torch.bool, device="npu")
        dual_out = torch.empty_like(query)
        ref_out = torch.empty_like(query)
        lse0 = torch.empty((0,), dtype=torch.float32, device="npu")
        lse1 = torch.empty((0,), dtype=torch.float32, device="npu")
        ref_lse0 = torch.empty((0,), dtype=torch.float32, device="npu")
        ref_lse1 = torch.empty((0,), dtype=torch.float32, device="npu")

        common_kwargs = dict(
            key=key,
            value=value,
            atten_mask=atten_mask,
            input_layout="TND",
            block_size=block_size,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            scale=scale,
            sparse_mode=3,
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False,
        )
        torch_npu.npu_fused_infer_attention_score.out(
            query=query[:split_tokens],
            block_table=block_table[:1],
            actual_seq_lengths=[split_tokens],
            actual_seq_lengths_kv=[4],
            out=[ref_out[:split_tokens], ref_lse0],
            **common_kwargs)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query[split_tokens:],
            block_table=block_table[1:2],
            actual_seq_lengths=[split_tokens],
            actual_seq_lengths_kv=[5],
            out=[ref_out[split_tokens:], ref_lse1],
            **common_kwargs)

        main_stream = torch.npu.current_stream()
        primary_stream = torch.npu.Stream()
        secondary_stream = torch.npu.Stream()
        torch.npu.dual_fused_infer_attention_score(
            main_stream,
            primary_stream,
            secondary_stream,
            query,
            key,
            value,
            dual_out,
            block_table_0=block_table[:1],
            block_table_1=block_table[1:2],
            actual_seq_lengths_0=[split_tokens],
            actual_seq_lengths_1=[split_tokens],
            actual_seq_lengths_kv_0=[4],
            actual_seq_lengths_kv_1=[5],
            split_start_0=0,
            split_graph_tokens_0=split_tokens,
            split_start_1=split_tokens,
            split_graph_tokens_1=split_tokens,
            atten_mask=atten_mask,
            softmax_lse_0=lse0,
            softmax_lse_1=lse1,
            num_heads=num_heads,
            scale=scale,
            block_size=block_size,
            num_key_value_heads=num_heads,
            sparse_mode=3,
            input_layout="TND",
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False)
        torch.npu.synchronize()

        torch.testing.assert_close(
            dual_out.cpu(), ref_out.cpu(), rtol=1e-2, atol=1e-2)

    @unittest.skipUnless(torch.npu.is_available(), "NPU is not available")
    def test_dual_fused_infer_attention_score_update_tnd_pa_replay_matches_two_out_calls(self):
        torch.npu.set_device(0)
        split_tokens = 1
        total_tokens = split_tokens * 2
        num_heads = 2
        head_dim = 128
        block_size = 128
        num_blocks = 2
        scale = 1.0

        query = torch.randn(total_tokens, num_heads, head_dim,
                            dtype=torch.bfloat16, device="npu")
        key = torch.randn(num_blocks, block_size, num_heads * head_dim,
                          dtype=torch.bfloat16, device="npu")
        value = torch.randn(num_blocks, block_size, num_heads * head_dim,
                            dtype=torch.bfloat16, device="npu")
        block_table = torch.tensor([[0], [1]], dtype=torch.int32, device="npu")
        ref_out = torch.empty_like(query)
        ref_lse0 = torch.empty((0,), dtype=torch.float32, device="npu")
        ref_lse1 = torch.empty((0,), dtype=torch.float32, device="npu")

        common_kwargs = dict(
            key=key,
            value=value,
            input_layout="TND",
            block_size=block_size,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            scale=scale,
            sparse_mode=0,
            pre_tokens=65535,
            next_tokens=65535,
            softmax_lse_flag=False,
        )
        torch_npu.npu_fused_infer_attention_score.out(
            query=query[:split_tokens],
            block_table=block_table[:1],
            actual_seq_lengths=[split_tokens],
            actual_seq_lengths_kv=[4],
            out=[ref_out[:split_tokens], ref_lse0],
            **common_kwargs)
        torch_npu.npu_fused_infer_attention_score.out(
            query=query[split_tokens:],
            block_table=block_table[1:2],
            actual_seq_lengths=[split_tokens],
            actual_seq_lengths_kv=[5],
            out=[ref_out[split_tokens:], ref_lse1],
            **common_kwargs)

        graph = torch.npu.NPUGraph()
        event = torch.npu.ExternalEvent()
        update_stream = torch.npu.Stream()
        output = None
        softmax_lse0 = None
        softmax_lse1 = None
        handle0 = None
        handle1 = None
        with torch.npu.graph(graph):
            stream = torch.npu.current_stream()
            output = torch.empty_like(query)
            softmax_lse0 = torch.empty(
                (0,), dtype=torch.float32, device="npu")
            softmax_lse1 = torch.empty(
                (0,), dtype=torch.float32, device="npu")
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query[:split_tokens],
                block_table=block_table[:1],
                actual_seq_lengths=[split_tokens],
                actual_seq_lengths_kv=[2],
                out=[output[:split_tokens], softmax_lse0],
                **common_kwargs)
            handle0 = torch.npu.graph_task_group_end(stream)
            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query[split_tokens:],
                block_table=block_table[1:2],
                actual_seq_lengths=[split_tokens],
                actual_seq_lengths_kv=[2],
                out=[output[split_tokens:], softmax_lse1],
                **common_kwargs)
            handle1 = torch.npu.graph_task_group_end(stream)

        dual_handle = torch.npu.make_dual_task_group_handle(handle0, handle1)
        with torch.npu.stream(update_stream):
            torch.npu.dual_fused_infer_attention_score_update(
                update_stream,
                dual_handle,
                query,
                key,
                value,
                output,
                block_table_0=block_table[:1],
                block_table_1=block_table[1:2],
                actual_seq_lengths_0=[split_tokens],
                actual_seq_lengths_1=[split_tokens],
                actual_seq_lengths_kv_0=[4],
                actual_seq_lengths_kv_1=[5],
                split_start_0=0,
                split_graph_tokens_0=split_tokens,
                split_start_1=split_tokens,
                split_graph_tokens_1=split_tokens,
                softmax_lse_0=softmax_lse0,
                softmax_lse_1=softmax_lse1,
                num_heads=num_heads,
                scale=scale,
                block_size=block_size,
                num_key_value_heads=num_heads,
                sparse_mode=0,
                input_layout="TND",
                pre_tokens=65535,
                next_tokens=65535,
                softmax_lse_flag=False)
            event.record(update_stream)

        graph.replay()
        torch.npu.synchronize()

        torch.testing.assert_close(
            output.cpu(), ref_out.cpu(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
